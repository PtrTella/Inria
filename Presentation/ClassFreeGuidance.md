Perfetto, procediamo con un’esposizione esaustiva e rigorosa del *classifier-free guidance* (CFG) come presentato in Ho & Salimans (2022). Ci soffermeremo su:

1. **Motivazioni e formalismo**
2. **Derivazione matematica dettagliata**
3. **Interpretazione geometrica e semantica**
4. **Implicazioni pratiche (e.g., text-to-image)**

---

## 1. Motivazione

Il *classifier guidance* (Dhariwal & Nichol, 2021) richiede un classificatore $C(x_t, y)$ per condizionare la generazione. Questo introduce vari problemi:

* È necessario addestrare un classificatore su immagini noiose $x_t$
* La guidance può essere percepita come un attacco avversario
* È difficile da generalizzare a classi aperte (e.g. prompt testuali arbitrari)

Il *classifier-free guidance* elimina il classificatore addestrando **un solo modello** di diffusione $\epsilon_\theta(x_t, t, y)$ in modo che possa funzionare **sia condizionato che non**.

---

## 2. Derivazione matematica dettagliata

### 2.1. Training del modello

Durante il training, ogni batch $(x_0, y)$ viene noised per ottenere $x_t$. Con probabilità $p$, il prompt $y$ viene *cancellato* (usando un token speciale come `NULL`). Si addestra quindi una rete $\epsilon_\theta(x_t, t, y)$ a predire il rumore originale $\epsilon$ usando l’obiettivo MSE:

$$
\mathcal{L}_\text{MSE} = \mathbb{E}_{x_0, t, \epsilon} \left[\left\|\epsilon_\theta(x_t, t, y) - \epsilon \right\|^2\right]
$$

Poiché la rete vede esempi sia condizionati che non, impara entrambe le distribuzioni.

### 2.2. Inference con guidance

Durante il sampling, il modello è chiamato due volte:

* **Condizionato**: $\epsilon_\theta(x_t, t, y)$
* **Non condizionato**: $\epsilon_\theta(x_t, t, \emptyset)$

Il termine interpolato è:

$$
\epsilon_{\text{guided}}(x_t, t, y) = (1 + w)\epsilon_\theta(x_t, t, y) - w\epsilon_\theta(x_t, t, \emptyset)
$$

Questa è un'interpolazione **lineare** tra i due predittori. Più formalmente, è una forma di *extrapolation* in direzione della condizione.

---

## 3. Interpretazione geometrica e semantica

Immaginiamo lo spazio delle previsioni del rumore come uno spazio vettoriale $\mathbb{R}^d$. Allora:

* $\epsilon_\theta(x_t, t, \emptyset)$ è la direzione “generica” del denoising
* $\epsilon_\theta(x_t, t, y)$ è la direzione “specifica al prompt”

La guidance amplifica la differenza:

$$
\epsilon_{\text{guided}} = \epsilon_{\text{uncond}} + w(\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})
$$

Quindi si naviga **lungo la direzione semantica** introdotta da $y$, con una forza controllata da $w$. Per $w = 0$, si ottiene campionamento non condizionato; per $w \to \infty$, si rischia *mode collapse* (perdita di diversità).

---

## 4. Applicazioni e implicazioni

### Text-to-Image

Nei modelli text-to-image (GLIDE, Imagen, Stable Diffusion):

* Il prompt $y$ è codificato da un encoder testuale (CLIP, T5, etc.)
* Il denoiser $\epsilon_\theta$ è una U-Net condizionata sul testo (via cross-attention)
* CFG è critico per ottenere immagini **semanticamente fedeli** al prompt

Ad esempio, con:

```python
guided_eps = (1 + w) * model(x_t, t, cond=y) - w * model(x_t, t, cond=None)
```

si ottiene una precisione semantica molto superiore rispetto all’uso del solo prompt.

---

Se vuoi, posso procedere con:

* una **derivazione analitica del gradiente del logaritmo della distribuzione condizionata**
* una **analogia bayesiana tra guidance e correzione di prior**
* o una **implementazione PyTorch** per chiarire ulteriormente.

Vuoi approfondire uno di questi aspetti?
