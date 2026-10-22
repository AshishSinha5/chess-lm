# Writing style

Write in simplified technical English, about 80% of the way to ASD-STE100.
It must read like a clear person, not like a machine.

Rules:
- One idea per sentence. Keep sentences to 20 words or fewer for instructions
  and 25 words or fewer for descriptions.
- Use the active voice. Write "The script reads the file", not "The file is read".
- Use short, common words. Write "use", not "utilize". Write "start", not "initiate".
- Use one word for one meaning. Do not swap synonyms for variety.
  Keep the same term for the same thing in the whole document.
- Use the simple tenses. Avoid the perfect and progressive forms when a simple
  form works.
- Write commands as direct instructions: "Run the test", "Set the value to 4".
- Do not use idioms, slang, or figurative language.
- Say what a thing is before you say what it does. Define each technical term once,
  at first use.
- Keep noun strings to 3 words or fewer.
- Put warnings and cautions before the step they apply to.
- Use numbered lists for steps in sequence. Use bullets for items with no order.

The 80% allowance. You may break a rule if it makes the text stiff or unclear.
Natural flow is more important than a strict rule. Never break a rule in a way that
adds ambiguity.

# Diagrams

Use a diagram when a picture shows the idea faster than text. This includes:
- data or request flow, and pipelines
- system architecture and component relationships
- state changes and sequences over time
- comparisons across more than 2 options and 2 attributes

Rules:
- Show the diagram first. Add 1 to 3 short sentences after it. Do not repeat in text
  what the diagram already shows.
- Choose the format by where the output will show:
  - Markdown or docs: Mermaid code block
  - Terminal or plain text: ASCII diagram
  - Web page or artifact: inline SVG
- Label every box and arrow. Use the same terms as the text and the code.
- Keep one diagram to about 10 nodes. If it needs more, split it into 2 diagrams.
- Do not use a diagram for a simple fact, a single step, or a short list.
- Write a one-line caption that says what the diagram shows.

# Log results after every step

Do this without being asked. A step is a finished data build, training run or sweep, speed test, evaluation, probe or figure set.
When a background run ends, do it at once, before you start the next step.

1. Run `bash scripts/build_results.sh`. It rebuilds `results/runs.csv`, `curves/`, `configs/` and the README table.
2. Add or update the entry in `results/EXPERIMENT_LOG.md`: date, command, run names, key numbers.
3. If the step led to a choice, add an entry to `DECISIONS.md` (see the decisions rule).
4. Update the Status and Next sections in `README.md`.
5. A new kind of result (token sweep, evaluation, probing) needs a script that writes a CSV into `results/`. Write the script first. Use the file names in `PLAN.md`, section Logging.
6. Do not commit or push unless the user asks. Tell the user which files changed and that they are not committed.

Never put checkpoints (`*.pt`) or W&B files in git.
