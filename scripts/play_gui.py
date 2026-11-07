"""Play chess against a trained model in your browser. No extra packages: it uses the Python standard library.

Use a model from Hugging Face (downloads one folder of sinha/chess-lm):
    python scripts/play_gui.py --size l
Use a local file:
    python scripts/play_gui.py --weights path/to/model.safetensors --config path/to/config.json
    python scripts/play_gui.py --ckpt /workspace/ckpt/size_m/final.pt
Then open http://127.0.0.1:8000. Click a piece, then click the square to move to.

The model picks the highest-scoring LEGAL move (greedy), or samples with --temperature above 0.
It reads only the move list, and it does not know the rules, so the program filters illegal moves."""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import torch

from chesslm.model.gpt import GPT, GPTConfig
from chesslm.tokenizers.uci_move import UCIMoveTokenizer

TOK = UCIMoveTokenizer.canonical()
MAX_TOKENS = 272  # model context length


def load_model(args):
    if args.ckpt:
        st = torch.load(args.ckpt, map_location="cpu")
        cfg, weights = st["model_cfg"], st["model"]
    else:
        from safetensors.torch import load_file

        if args.weights:
            wpath, cpath = Path(args.weights), Path(args.config or Path(args.weights).with_name("config.json"))
        else:
            from huggingface_hub import snapshot_download

            root = snapshot_download(args.repo, allow_patterns=[f"{args.size}/*"])
            wpath, cpath = Path(root) / args.size / "model.safetensors", Path(root) / args.size / "config.json"
        cfg, weights = json.load(open(cpath))["model_cfg"], load_file(wpath)
    model = GPT(GPTConfig(**cfg))
    model.load_state_dict(weights)
    return model.eval()


class Game:
    def __init__(self, model, temperature):
        self.model, self.temperature = model, temperature
        self.new("white")

    def new(self, human):
        self.human = chess.WHITE if human == "white" else chess.BLACK
        self.board = chess.Board()
        self.moves = []
        self.error = ""
        if self.board.turn != self.human:
            self.model_move()

    @torch.no_grad()
    def model_move(self):
        if self.board.is_game_over() or len(self.moves) + 1 >= MAX_TOKENS:
            return
        ids = torch.tensor([[TOK.bos] + [TOK.stoi[m] for m in self.moves]])
        logits = self.model(ids)[0][0, -1].float()
        legal = [m.uci() for m in self.board.legal_moves]
        scores = torch.stack([logits[TOK.stoi[u]] for u in legal])
        if self.temperature > 0:
            pick = int(torch.multinomial(torch.softmax(scores / self.temperature, 0), 1))
        else:
            pick = int(scores.argmax())
        self.push(legal[pick])

    def push(self, uci):
        self.board.push_uci(uci)
        self.moves.append(uci)

    def human_move(self, uci):
        self.error = ""
        move = chess.Move.from_uci(uci)
        if self.board.turn != self.human or self.board.is_game_over():
            self.error = "It is not your turn."
        elif move not in self.board.legal_moves:
            piece = self.board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN and chess.square_rank(move.to_square) in (0, 7):
                move = chess.Move.from_uci(uci + "q")  # promote to a queen
        if not self.error and move not in self.board.legal_moves:
            self.error = "Illegal move."
        if not self.error:
            self.push(move.uci())
            self.model_move()

    def undo(self):
        """Take back the last full turn (your move and the reply)."""
        n = 2 if self.board.turn == self.human and len(self.moves) >= 2 else 1
        for _ in range(min(n, len(self.moves))):
            self.board.pop()
            self.moves.pop()
        if self.board.turn != self.human:
            self.model_move()

    def state(self):
        b = self.board
        status = ""
        if b.is_checkmate():
            status = "Checkmate. " + ("You win." if b.turn != self.human else "The model wins.")
        elif b.is_game_over():
            status = "Draw (" + (b.outcome().termination.name.lower().replace("_", " ") if b.outcome() else "") + ")."
        elif len(self.moves) + 1 >= MAX_TOKENS:
            status = "Game stopped: the model's context is full."
        elif b.is_check():
            status = "Check."
        over = b.is_game_over() or len(self.moves) + 1 >= MAX_TOKENS
        legal = {}
        for m in b.legal_moves:
            legal.setdefault(chess.square_name(m.from_square), []).append(chess.square_name(m.to_square))
        return {"fen": b.fen(), "human": "white" if self.human else "black", "turn": "white" if b.turn else "black",
                "legal": legal if b.turn == self.human and not over else {},
                "last": self.moves[-1] if self.moves else "", "moves": self.moves, "status": status, "error": self.error}


PAGE = """<!doctype html><meta charset=utf-8><title>Chess LM</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#222;color:#eee;display:flex;gap:28px;flex-wrap:wrap}
#board{display:grid;grid-template-columns:repeat(8,64px);grid-template-rows:repeat(8,64px);border:3px solid #555}
.sq{width:64px;height:64px;display:flex;align-items:center;justify-content:center;font-size:46px;cursor:pointer;position:relative;user-select:none}
.l{background:#f0d9b5;color:#000}.d{background:#b58863;color:#000}
.sel{outline:4px solid #4a90e2;outline-offset:-4px}.last{box-shadow:inset 0 0 0 100px rgba(255,235,59,.35)}
.dot::after{content:"";width:18px;height:18px;border-radius:50%;background:rgba(0,0,0,.3);position:absolute}
.w{color:#fff;text-shadow:0 0 2px #000,0 0 2px #000,0 0 2px #000}.b{color:#000}
#side{max-width:280px}button{margin:4px 4px 4px 0;padding:6px 12px}#st{min-height:1.4em;font-weight:600}#err{color:#f88;min-height:1.4em}
#mv{font-family:monospace;line-height:1.5}
</style>
<div><div id=board></div></div>
<div id=side><h2>Chess LM</h2><div id=st></div><div id=err></div>
<div><button onclick="newg('white')">New game: white</button><button onclick="newg('black')">New game: black</button>
<button onclick="undo()">Undo</button></div><p>Moves:</p><div id=mv></div></div>
<script>
const G={p:"♟",n:"♞",b:"♝",r:"♜",q:"♛",k:"♚"};let S=null,sel=null;
async function call(path,body){const r=await fetch(path,{method:"POST",body:JSON.stringify(body||{})});S=await r.json();draw()}
function newg(c){sel=null;call("/new",{human:c})}function undo(){sel=null;call("/undo")}
function click(sq){
 if(sel&&S.legal[sel]&&S.legal[sel].includes(sq)){const m=sel+sq;sel=null;call("/move",{uci:m});return}
 sel=S.legal[sq]?sq:null;draw()}
function draw(){
 const rows=S.fen.split(" ")[0].split("/"),flip=S.human==="black",el=document.getElementById("board");el.innerHTML="";
 const last=S.last?[S.last.slice(0,2),S.last.slice(2,4)]:[];
 for(let r=0;r<8;r++)for(let c=0;c<8;c++){
  const rr=flip?7-r:r,cc=flip?7-c:c;let file=0,piece="";
  for(const ch of rows[rr]){if(/\\d/.test(ch))file+=+ch;else{if(file===cc)piece=ch;file++}}
  const name="abcdefgh"[cc]+(8-rr),d=document.createElement("div");
  d.className="sq "+((rr+cc)%2?"d":"l")+(sel===name?" sel":"")+(last.includes(name)?" last":"")+(sel&&S.legal[sel]&&S.legal[sel].includes(name)?" dot":"");
  if(piece){const s=document.createElement("span");s.textContent=G[piece.toLowerCase()];s.className=piece===piece.toUpperCase()?"w":"b";d.appendChild(s)}
  d.onclick=()=>click(name);el.appendChild(d)}
 document.getElementById("st").textContent=S.status||(S.turn===S.human?"Your move.":"The model is thinking...");
 document.getElementById("err").textContent=S.error;
 document.getElementById("mv").textContent=S.moves.map((m,i)=>(i%2?"":(i/2+1)+". ")+m).join(" ")}
call("/state")
</script>"""


def make_handler(game):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype="application/json"):
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send(PAGE, "text/html")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
            try:
                if self.path == "/new":
                    game.new(body.get("human", "white"))
                elif self.path == "/move":
                    game.human_move(body["uci"])
                elif self.path == "/undo":
                    game.undo()
            except Exception as e:  # keep the server alive on bad input
                game.error = f"Error: {e}"
            self.send(json.dumps(game.state()))

    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="l", choices=["xs", "s", "m", "l"], help="model size to download from --repo")
    ap.add_argument("--repo", default="sinha/chess-lm")
    ap.add_argument("--weights", help="local model.safetensors")
    ap.add_argument("--config", help="local config.json (default: next to --weights)")
    ap.add_argument("--ckpt", help="local training checkpoint (.pt)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = best move; above 0 = sample")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    torch.set_num_threads(2)
    model = load_model(a)
    print(f"Model ready. Open http://{a.host}:{a.port}", file=sys.stderr)
    ThreadingHTTPServer((a.host, a.port), make_handler(Game(model, a.temperature))).serve_forever()
