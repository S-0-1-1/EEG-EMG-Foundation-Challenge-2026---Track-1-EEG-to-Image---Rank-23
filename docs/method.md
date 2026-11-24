# Method

This note describes the model, the training objective, the two augmentations, the two-stage training and the ensemble. Each mechanism is first written as an equation and then explained in words. All numbers quoted here are means over 20 held-out people (five folds of four people) on the 200 public test images, unless stated otherwise.

## 1. Task and score

A trial $x \in \mathbb{R}^{32 \times 120}$ is one second of electroencephalography (EEG) on 32 channels at 120 Hz, from 0.2 s before to 0.8 s after the onset of an image. The model returns a vector $p(x) \in \mathbb{R}^{1536}$.

Every image $m$ has a fixed target vector $g_m$ computed by the organizers from DINOv2-giant. In release 0.4.0 of their loader, the target is the mean over tokens of the layer at two thirds of the depth of the network, for images of side 518.

For a person $s$ and an image $m$ seen $n$ times, the score first averages the predictions and then ranks the gallery:

```math
\bar p_{s,m} = \frac{1}{n} \sum_{i \,:\, s_i = s,\; m_i = m} p(x_i), \qquad
\text{rank of } m = \#\Big\{ j : \cos(\bar p_{s,m},\, g_j) > \cos(\bar p_{s,m},\, g_m) \Big\} . \tag{1}
```

The query counts as correct when this rank is below five. The score is the share of correct queries.

Two consequences follow from equation (1). Only the direction of the output matters, so all models are trained with cosine similarities. And the predictions of several presentations are added before the ranking, so the output of every trial is given unit length, which gives each presentation the same weight.

## 2. From a trial to the input of the encoder

The loader of the organizers delivers trials that are already preprocessed. In release 0.4.0 the signal is filtered between 0.1 and 75 Hz with notch filters at 50 and 60 Hz, resampled to 120 Hz, corrected with the 0.2 s before the image as baseline, scaled per channel with a robust scaler and limited to the interval from -20 to 20. The encoder was pretrained on signals at 200 Hz in units of 0.1 mV. The trial is therefore resampled and rescaled:

```math
\tilde x = 0.1349 \cdot R_{120 \rightarrow 200}(x) \in \mathbb{R}^{32 \times 200} . \tag{2}
```

$R$ is a band-limited resampling. The factor is $1.349 \times 0.1$: a robust scaler divides by the interquartile range, which is 1.349 standard deviations for a normal variable, so the first factor gives the channels a standard deviation close to one and the second factor brings them to the amplitude range of the pretraining data.

## 3. Network

**Encoder.** LaBraM (Large Brain Model) cuts every channel into patches of 200 samples, embeds each patch with a small temporal convolution, adds an embedding of the electrode and an embedding of the time position, and applies 12 transformer layers of width 200. One second at 200 Hz is exactly one patch, so a trial becomes 32 tokens $h_1, \dots, h_{32} \in \mathbb{R}^{200}$, one per electrode. The electrode embeddings are selected by channel name, which lets the pretrained weights be used on the montage of the headset. The encoder has 5.8 million weights.

**Head.** The tokens keep their identity up to the output:

```math
f(x) = W_2 \, \mathrm{Dropout}\Big( \mathrm{LayerNorm}\big( [\, \phi(h_1); \dots; \phi(h_{32}) \,] \big) \Big) + b_2,
\qquad \phi(h) = \mathrm{GELU}(W_1 h + b_1) \in \mathbb{R}^{64} . \tag{3}
```

Each token is reduced from 200 to 64 values by the same linear layer, the 32 results are joined into a vector of 2048 values, and one linear layer gives the 1536 outputs. The dropout rate is 0.3. The head has 3.2 million weights. Joining the tokens instead of averaging them keeps the information of where on the scalp a pattern was seen.

All weights of the encoder and of the head are trained together from the first step.

## 4. Training objective

Let $z_i = f(x_i) / \lVert f(x_i) \rVert$ be the unit output for trial $i$ of a batch of $B$ trials, $m_i$ its image, $u_m = g_m / \lVert g_m \rVert$ the unit target of image $m$, and $M_B$ the images that are present in the batch.

**Contrast inside the batch.** Each trial has to recognize its own image among the images of the batch:

```math
L_{\text{con}} = - \frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\alpha \, z_i^\top u_{m_i})}{\sum_{m \in M_B} \exp(\alpha \, z_i^\top u_m)} . \tag{4}
```

The scale $\alpha$ is learned, starts at $1/0.07$ and is limited to 100. Two trials of the same image in one batch share one column, so they are never treated as negatives of each other.

**Cosine term on the centered target.** The targets of all images share a large common component. Let $\bar u$ be the mean of the unit targets of the training images. The centered target is

```math
c_m = \frac{u_m - \bar u}{\lVert u_m - \bar u \rVert} , \tag{5}
```

and the full loss is

```math
L = L_{\text{con}} + \lambda \, \frac{1}{B} \sum_{i=1}^{B} \big( 1 - z_i^\top c_{m_i} \big), \qquad \lambda = 0.5 . \tag{6}
```

The contrast only compares images that happen to be in the batch. The cosine term gives every trial a target of its own. Without centering, this term mostly asks the output to point to the common component $\bar u$, which carries no information for the ranking: the single-trial top-5 is then 0.0680, below the 0.0747 of the contrast alone. With the centered target of equation (5) it is 0.0757. Equation (5) is also the direction in which image $m$ differs from the other images, which is what the ranking of equation (1) uses.

## 5. Optimization

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight decay 0.01 |
| Learning rate | 0.0003 for the encoder, 0.001 for the head and for $\alpha$ |
| Schedule | linear warm-up over 300 steps, then cosine decay to zero |
| Batch | 256 trials |
| Length | 12 epochs, each one a new random draw of 400,000 trials (1,562 steps) |
| Gradient | norm clipped at 1.0 |
| Arithmetic | half precision on GPU |
| Model kept | the last epoch, no validation set |

The training is short on purpose. A run of 30 epochs without augmentation has its best held-out score after six to nine epochs (0.076) and ends at 0.060: the model specializes on its training people.

## 6. Person transport

The largest obstacle is the change of person. A model trained on 16 people sees 16 different mappings from brain activity to the 32 electrodes. The first augmentation produces trials whose spatial statistics lie between those of two real people.

**Covariance of a person.** For each training person $a$, 3,000 trials are drawn. With $\check x_{n,t} \in \mathbb{R}^{32}$ the sample $t$ of trial $n$ after removal of the mean of each channel over the trial, the covariance is estimated and slightly loaded on its diagonal:

```math
S_a = \frac{1}{N T} \sum_{n=1}^{N} \sum_{t=1}^{T} \check x_{n,t} \, \check x_{n,t}^\top, \qquad
C_a = S_a + \epsilon \, \frac{\mathrm{tr}(S_a)}{32} \, I, \qquad \epsilon = 10^{-3} . \tag{7}
```

**Transport map.** The linear map that carries a centered normal distribution of covariance $C_a$ onto one of covariance $C_b$ with the least mean squared displacement is

```math
T_{ab} = C_a^{-1/2} \big( C_a^{1/2} \, C_b \, C_a^{1/2} \big)^{1/2} C_a^{-1/2}, \qquad T_{ab} \, C_a \, T_{ab}^\top = C_b . \tag{8}
```

$T_{ab}$ is a symmetric matrix of size 32 by 32 that recombines the electrodes. It is computed once for every ordered pair of training people.

**Augmented trial.** With probability 0.5, a trial $x$ of person $a$ is replaced by

```math
x' = \big( (1 - t) \, I + t \, T_{ab} \big) \, x, \qquad t \sim \mathcal{U}(0, 1), \tag{9}
```

where $b$ is another training person chosen at random. The same matrix is applied to every time sample, so the time course of the response is not changed. For $t = 0$ the trial is unchanged; for $t = 1$ its spatial covariance is that of person $b$; in between it follows the shortest path between the two covariances in the sense of optimal transport. The image and the target stay those of the original trial.

**Time shift.** Together with the transport, and also with probability 0.5, the trial is shifted by one or two samples (8 or 17 ms) to the left or to the right. The sample at the border is repeated.

Effect: the single-trial top-5 goes from 0.0757 to 0.0784.

## 7. Same-image blending

Every training image is seen by all training people, several times each. The second augmentation uses this. For trial $x_i$ of image $m$, a second trial $x_j$ of the same image is drawn at random among all training trials of $m$, and with probability 0.5

```math
x' = x_i + \mu \, (x_j - x_i), \qquad \mu \sim \mathcal{U}(0, 0.5) . \tag{10}
```

Both trials contain the response to the same image, so this response is kept. What differs between them, the noise of the single trial and the traits of the person, is reduced for $x_i$ and replaced in the same proportion by that of $x_j$. The target is unchanged. Because $\mu \le 0.5$, the first trial always dominates.

The order inside a training step is: blending, then the transport of equation (9) with the map of the person of $x_i$, then the time shift.

Effect: 0.0784 becomes 0.0802 for one presentation, and 0.2473 becomes 0.2658 when all presentations are averaged.

## 8. Start from a model trained on THINGS-EEG2

THINGS-EEG2 shows the same images to 10 other people with a research cap of 63 electrodes. Its montage contains the 32 electrodes of the headset.

**Stage one.** The same network is trained on THINGS-EEG2, on these 32 electrodes selected by name and on the same window of 120 samples. Since the electrodes and the window are the same, the complete network, encoder and head, can be reused.

**Stage two.** The training on Alljoined-1.6M starts from the stage-one weights instead of the published LaBraM weights. Everything else is unchanged.

The benefit depends on the augmentation:

| Recipe of stage two | usual start | THINGS-EEG2 start |
|---|---|---|
| no augmentation | 0.0757 | 0.0703 |
| person transport | 0.0784 | 0.0788 |
| person transport and blending | 0.0802 | 0.0832 |

Without augmentation the start lowers the score among 200 images. With person transport alone, the model started from THINGS-EEG2 is best after five epochs (0.0814) and ends at 0.0788, because it specializes on its 16 training people. With both augmentations it is at 0.0766 after the first epoch and at 0.0832 at the end: the gain is kept, and this is the best single model. In the gallery of 16,740 images the start helps in all three rows (top-50 with one presentation: 0.0086 to 0.0087, 0.0088 to 0.0103, 0.0092 to 0.0104).

## 9. Ensemble

The submission averages the unit outputs of $K = 11$ models and gives the result unit length:

```math
p(x) = \frac{\sum_{k=1}^{K} z^{(k)}(x)}{\big\lVert \sum_{k=1}^{K} z^{(k)}(x) \big\rVert}, \qquad z^{(k)}(x) = \frac{f_k(x)}{\lVert f_k(x) \rVert} . \tag{11}
```

No weights are fitted and nothing is estimated on the test trials: each trial is predicted alone, which respects the rule that no state is kept between calls.

The eleven models are two runs for each of five recipes (both augmentations, person transport alone, each with both starts, and the loss of equation (6) without augmentation from the THINGS-EEG2 start) and one run of the contrast of equation (4) alone. On held-out people, an ensemble with one model per recipe reaches 0.0862 with one presentation, against 0.0832 for the best single model.

## 10. Evaluation on held-out people

The 20 people of Alljoined-1.6M are split into five folds of four. A model is trained on the training images of 16 people and predicts the trials of the 200 test images for the four others. No image and no person of a query has been seen by the model that answers it.

A query is built from the presentations of one image to one person, as in equation (1), in three ways: one presentation, two presentations (random disjoint pairs, at most 40 pairs for one person and one image) and all presentations. The same queries are ranked in two galleries: the 200 test images (top-1 and top-5) and all 16,740 public images (top-5 and top-50). The accuracy is computed for each person and then averaged over the 20 people.

The second gallery matters because the sealed gallery is large. A change that helps among 200 images but not among 16,740 images was not kept.
