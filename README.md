# Bonsai-Life

A particle-life breeder that runs in your browser, with [Ternary-Bonsai-2-2B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf) as the resident naturalist.

Four universes evolve side by side. A deterministic `alive_score` (the fraction of particles in clumps times how fast those clumps move together) picks each generation's winner. Bonsai looks at a snapshot of the winner, gives it a species name and field notes, and proposes 2-4 attraction-matrix edits toward your goal.

| Slot | What it is |
|---|---|
| A | the champion, unchanged |
| B | the champion plus Bonsai's proposed edits |
| C, D | the champion plus random Gaussian mutations |

The "AI slot wins" counter shows how often Bonsai's edit beats random mutation. Click a quadrant to crown it yourself. Click a matrix cell to raise the champion's attraction, or shift-click to lower it.

## Play it in the browser

Open the hosted page and press **Wake Bonsai in this browser**. Bonsai 2B runs on your GPU through WebGPU, using PrismML's engine from their Talk app (`web/vendor/bonsai.mjs`, Apache-2.0). The first visit downloads about 794 MB of weights from Hugging Face; later visits load them from the browser cache. Without WebGPU, the sim still runs and an offline naturalist writes the notes.

Any static HTTPS host works: serve the `web/` folder.

## Brain explorer (`web/neurons.html`)

A second page that opens up the model itself, which only works because Bonsai runs on your own GPU:

- **Neuron cloud.** 192 neurons from each of the 24 MLP layers, copied off the GPU after every layer for every word, drawn as a 3D cloud. Dots light up by how far each neuron is from its usual level, and an orange path follows the strongest firing up the stack. Spot-checked against PrismML's llama.cpp (correlation 0.9999).
- **Word map.** Each word's internal state at a chosen layer, squashed to 2D.
- **Live odds.** The top 5 candidates and their probabilities at every word.
- **Surgery.** Switch off the token mixer or the MLP of any layer by zeroing the f16 scale shared by each group of 128 ternary weights. Edits are made in GPU memory and restored bit-exactly. A reference loss (nats per token on a fixed paragraph of easy facts) tracks how much dumber it got, and the text is rewritten so the cloud shows the damage.

Open `neurons.html?stub=1` to try the UI with a tiny fake model and no GPU. `node tests/run-tests.cjs` runs the weight-surgery checks on a software WebGPU device (bit-exact restore, only the intended scales change). The older `brain.html` (layer stack, lesion scan, damage sweep, a steering game) still works but is no longer linked.

## Run it locally with llama-server

```bash
./start.sh            # downloads PrismML's llama.cpp build + PQ2_0 weights + Q8_0 mmproj, starts everything
./start.sh --no-model # simulation + offline naturalist only
```

Then open http://localhost:8000. On Apple Silicon the model runs on Metal. On Linux it uses CUDA if `nvidia-smi` exists, and the CPU otherwise.

Overrides: `MODEL_PATH`, `MMPROJ_PATH`, `LLAMA_SERVER`, `PORT`, `LLAMA_PORT`, `PRISM_TAG`.

## How it fits together

- `web/index.html`: the simulation (spatial-hash particle life, 6 species, 1,100 particles per universe), rendering, the evolution loop and the UI, with no dependencies.
- `server.py`: serves the page and proxies `/api/naturalist` to llama-server with `enable_thinking: false` and a JSON schema, then validates the edits.
- If Bonsai takes longer than 8 s, the next generation starts with a random B, and Bonsai's edits are hot-swapped into B as soon as they arrive.

## Secret messages (`web/secret.html`)

Type a short secret and Bonsai writes an ordinary tweet whose token choices encode it. Paste the tweet into the same page on any device and Bonsai reads it back.

- A step carries a digit only when the top 2 to 4 tokens are all plausible and clearly ahead of the rest; the digit is the chosen token's position among them, sorted by token id. The encoder only uses steps that clear every threshold by a margin (`delta`), so small logit differences between GPUs and browsers don't change what the reader sees.
- Payload: fixed Huffman code for the secret plus a CRC16, read as one mixed-radix integer. The reader tries each cover topic's prompt and keeps the one whose checksum matches.
- The Lab tab records a logit fingerprint (compare two devices' reports), sweeps capacity settings and runs encode/decode round trips. URL params override settings, e.g. `secret.html?delta=0.4&W=4`; `?stub` runs a fake model.
- Tests: `node test/stego.test.mjs` (stub model with injected logit noise; includes a no-margin control that fails).

Prior art: Ziegler et al. 2019 (arithmetic-coding stego), Meteor (2021), Discop (2023), Steganeur. This version's twist is running fully in the browser and surviving cross-device numerics.
