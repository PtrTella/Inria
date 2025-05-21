Il moto di Langevin svolge un ruolo fondamentale nei modelli di diffusione, specialmente nel contesto dello *score-based generative modeling*, come illustrato da Song & Ermon (2020), e trova un parallelo nei modelli DDPM (Denoising Diffusion Probabilistic Models) di Ho et al. (2020). Procediamo con un'analisi **matematica dettagliata** del suo uso nei modelli di diffusione e nel processo di inferenza.

---

## 1. **Dinamica di Langevin: Definizione Matematica**

Il *moto di Langevin* è una discretizzazione stocastica della dinamica di gradient ascent con rumore:

$$
\tilde{x}_{t+1} = \tilde{x}_t + \frac{\epsilon}{2} \nabla_x \log p(x)|_{x=\tilde{x}_t} + \sqrt{\epsilon} z_t,\quad z_t \sim \mathcal{N}(0, I)
$$

* $\epsilon > 0$: passo temporale.
* $\nabla_x \log p(x)$: **score function**, ovvero il gradiente del logaritmo della densità dati.
* $z_t$: rumore gaussiano standard.

Per $\epsilon \to 0$ e $T \to \infty$, $\tilde{x}_T \sim p(x)$ sotto condizioni regolari.

---

## 2. **Sampling nei Modelli Score-based (NCSN)**

Song & Ermon (2020) utilizzano **annealed Langevin dynamics** per generare campioni da una distribuzione dati sconosciuta $p_{\text{data}}(x)$, attraverso una rete neurale $s_\theta(x, \sigma) \approx \nabla_x \log q_\sigma(x)$, dove $q_\sigma$ è una versione "noiseata" di $p_{\text{data}}$.

### **Sampling Procedure**:

Dato un set decrescente di livelli di rumore $\{\sigma_1 > \dots > \sigma_L\}$:

$$
x^{(0)} \sim \mathcal{N}(0, I)
$$

$$
x^{(t+1)} = x^{(t)} + \frac{\alpha_i}{2} s_\theta(x^{(t)}, \sigma_i) + \sqrt{\alpha_i} z_t
$$

* Ogni livello $\sigma_i$ induce una distribuzione $q_{\sigma_i}(x)$.
* Il parametro $\alpha_i = \epsilon \cdot \frac{\sigma_i^2}{\sigma_L^2}$ controlla il passo a ogni livello.

Questo approccio mitiga le difficoltà imposte dalla *manifold hypothesis* e migliora il *mixing* tra le modalità della distribuzione, evitando che le dinamiche rimangano "bloccate" in regioni di bassa densità.

---

## 3. **Relazione con Modelli di Diffusione (DDPM)**

Nei DDPM (Ho et al. 2020), la dinamica di campionamento è formalmente simile a un’integrazione inversa stocastica simile a Langevin, anche se in forma più controllata:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

* $\epsilon_\theta$ predice il rumore (invece del gradiente log-densità esplicitamente).
* È matematicamente equivalente a denoising score matching su più livelli di rumore.
* La procedura di campionamento assomiglia strettamente alla dinamica di Langevin "annealed", poiché ogni passo corrisponde a una transizione condizionata Gaussianamente dal passo precedente.

---

## 4. **Aspetti Termodinamici e Fokker-Planck**

Sohl-Dickstein et al. (2015) mostrano che le dinamiche di diffusione e la loro inversione si collegano all’equazione di Fokker-Planck, con una derivazione formale che fa emergere Langevin come soluzione stocastica:

$$
dx = f(x, t) dt + g(t) dW_t
$$

dove $f$ è il drift (score), $g$ la scala del rumore, e $W_t$ un moto browniano. In equilibrio, questa SDE converge a $p_{\text{data}}(x)$.

---

## 5. **Considerazioni Pratiche**

* **Inaccurate score in low-density**: In regioni a bassa densità dati, i gradienti stimati sono rumorosi e Langevin può richiedere molti passi o non mescolare correttamente.
* **Annealed Dynamics**: Mitiga il problema grazie a un'inizializzazione in distribuzioni "rumorose" e una progressiva raffinazione.
* **Inferenza**: In modelli condizionati (e.g. inpainting, text2image), il gradiente può includere anche il termine di likelihood da un classificatore per guidare il campionamento (classifier guidance).

---

## Conclusione

Il moto di Langevin è fondamentale per i modelli di generazione basati su score function. Nei modelli di diffusione come DDPM o score-based NCSN, le dinamiche annealed di Langevin permettono un processo di campionamento efficace e teoricamente fondato, capace di superare le limitazioni di tecniche tradizionali (GAN, VAE) in termini di stabilità, diversità e fedeltà.
