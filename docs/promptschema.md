# Prompt Schema

We use OpenAI's prompt structure when interacting. 

```json
[
     {
            "messages":[
                {
                    "role":"system",
                    "content":"You will write a human-like abstract for the paper with the following title. The user will give you an example abstract to mimic its style. Output only the new abstract, nothing else.\nPaper Title: High Spatial Resolution HST\/NICMOS Observations of Markarian 231"
                },
                {
                    "role":"user",
                    "content":"The present spatial distribution of galaxies in the Universe is non-Gaussian, with 40% skewness in 50 Mpc\/h spheres, and remarkably little is known about the information encoded in it about cosmological parameters beyond the power spectrum. In this work we present an attempt to bridge this gap by studying the bispectrum, paying particular attention to a joint analysis with the power spectrum and their combination with CMB data. We address the covariance properties of the power spectrum and bispectrum including the effects of beat coupling that lead to interesting cross-correlations, and discuss how baryon acoustic oscillations break degeneracies. We show that the bispectrum has significant information on cosmological parameters well beyond its power in constraining galaxy bias, and when combined with the power spectrum is more complementary than combining power spectra of different samples of galaxies, since non-Gaussianity provides a somewhat different direction in parameter space. In the framework of flat cosmological models we show that most of the improvement of adding bispectrum information corresponds to parameters related to the amplitude and effective spectral index of perturbations, which can be improved by almost a factor of two. Moreover, we demonstrate that the expected statistical uncertainties in sigma8 of a few percent are robust to relaxing the dark energy beyond a cosmological constant."
                }
            ],
            "temperature":1.9093836992,
            "top_p":0.0121978974
    }
]
```