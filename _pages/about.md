---
permalink: /
title: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<!-- This is the front page of a website that is powered by the [academicpages template](https://github.com/academicpages/academicpages.github.io) and hosted on GitHub pages. [GitHub pages](https://pages.github.com) is a free service in which websites are built and hosted from code and data stored in a GitHub repository, automatically updating when a new commit is made to the respository. This template was forked from the [Minimal Mistakes Jekyll Theme](https://mmistakes.github.io/minimal-mistakes/) created by Michael Rose, and then extended to support the kinds of content that academics have: publications, talks, teaching, a portfolio, blog posts, and a dynamically-generated CV. You can fork [this repository](https://github.com/academicpages/academicpages.github.io) right now, modify the configuration and markdown files, add your own PDFs and other content, and have your own site for free, with no ads! An older version of this template powers my own personal website at [stuartgeiger.com](http://stuartgeiger.com), which uses [this Github repository](https://github.com/staeiou/staeiou.github.io). -->

About me
======
As a current PhD student, I have the honor of being guided by Professor Shugong Xu, a distinguished IEEE Fellow (https://www.researchgate.net/profile/Shugong-Xu-2). My research is deeply rooted in the field of speech technology, encompassing speaker recognition, keyword recognition, speech synthesis, zero-shot voice cloning, federated learning, transfer learning, and few-shot learning. Presently, my active research interests include exploring audio foundation models, speech synthesis, and multi-modal learning theory. I possess a profound interest in the applications and development trends of machine learning technologies within the speech and audio domain.

My Active Research Projects
--------
**StableTTS: Towards Fast Denoising Acoustic Decoder for Text to Speech Synthesis with Consistency Flow Matching**

Current state-of-the-art text-to-speech (TTS) systems predominantly utilize denoising-based acoustic decoders, which are known for their superior performance. In this study, we introduce an efficient TTS system that incorporates Consistency Flow Matching denoising training and a Diffusion Transformer (DiT) block. This training approach not only significantly enhances the cost-efficiency and operational efficiency of flow matching blocks in existing systems but also maintains high performance levels. Additionally, the DiT block, which aligns with the latest advancements in denoising approaches, simplifies the training process. Our comprehensive evaluations, conducted on an in-domain dataset and against various denoising-based TTS systems, affirm the denoising efficiency for our proposed system.
<p align="center">
  <img src="http://zhiyongchenGREAT.github.io/images/stable.png" alt="stable" width="50%" />
</p>

Paper Link: Coming soon

**ZS-TTS/Voice Cloning and Synthesis: Optimizing Feature Fusion for Improved Zero-shot Adaptation in Text-to-Speech Synthesis**

A primary challenge in VC is maintaining speech quality and speaker similarity with limited reference data for a specific speaker. However, existing VC systems often rely on naive combinations of embedded speaker vectors for speaker control, which compromises the capture of speaking style, voice print, and semantic accuracy. To overcome this, we introduce the Two-branch Speaker Control Module (TSCM), an novel and highly adaptable voice cloning module designed to precisely processing speaker or style control for a target speaker. Our method uses an advanced fusion of local-level features from a Gated Convolutional Network (GCN) and utterance-level features from a Gated Recurrent Unit (GRU) to enhance speaker control. We demonstrate the effectiveness of TSCM by integrating it into advanced TTS systems like FastSpeech 2 and VITS architectures, significantly optimizing their performance. Experimental results show that TSCM enables accurate voice cloning for a target speaker with minimal data through both zero-shot or few-shot fine-tuning of pre-trained TTS models. Furthermore, our TSCM based VITS (TSCM-VITS) showcases superior performance in zero-shot scenarios compared to existing state-of-the-art VC systems, even with basic dataset configurations. Our method's superiority is validated through comprehensive subjective and objective evaluations.

<p align="center">
  <img src="http://zhiyongchenGREAT.github.io/images/tscm.png" alt="TSCM" width="50%" />
</p>

[Project Website](https://great-research.github.io/tsct-tts-demo/) 

[Research paper: TSCM-VITS](http://zhiyongchenGREAT.github.io/files/s13636-024-00351-9.pdf)

**Emotional Style Control TTS: StyleFusion TTS--Multimodal Style-control and Enhanced Feature Fusion for Zero-shot Text-to-speech Synthesis**

We introduce StyleFusion-TTS, a prompt and/or audio ref- erenced, style- and speaker-controllable, zero-shot text-to-speech (TTS) synthesis system designed to enhance the editability and naturalness of current research literature. We propose a general front-end encoder as a compact and effective module to utilize multimodal inputs—including text prompts, audio references, and speaker timbre references—in a fully zero-shot manner and produce disentangled style and speaker control embeddings. Our novel approach also leverages a hierarchical conformer structure for the fusion of style and speaker control embeddings, aiming to achieve optimal feature fusion within the current advanced TTS ar- chitecture. StyleFusion-TTS is evaluated through multiple metrics, both subjectively and objectively. The system shows promising performance across our evaluations, suggesting its potential to contribute to the ad- vancement of the field of zero-shot text-to-speech synthesis. 

<p align="center">
  <img src="http://zhiyongchenGREAT.github.io/images/style_fusion.png" alt="style_fusion" width="50%" />
</p>

[Project Website](https://srplplus.github.io/StyleFusionTTS-demo/) 

[Research paper: StyleFusion](http://zhiyongchenGREAT.github.io/files/StyleTTS_PRCV24-11.pdf)

**MM-KWS: Personalized User-Defined Keyword Spotting and Open-set Speaker Identification in Household Environments**

We introduce Personalized User-Defined Keyword Spotting (PUKWS), a novel pipeline specifically designed for enhancing household environments by integrating user-defined keyword spotting (KWS) with open-set speaker identification (SID) into a cascading dual sub-system structure. For KWS, we present multi-modal user-defined keyword spotting (M-UDKWS), a novel approach that leverages multi-modal prompts for text-audio multimodal enrollment, and optimizes phonetic and semantic feature extraction to synergize text and audio modalities. This innovation not only stabilizes detection by reducing mismatches between query audio and support text embeddings but also excels in handling potentially confusing keywords. For open-set SID, we adopt advanced open-set learning techniques to propose speaker reciprocal points learning (SRPL), addressing the significant challenge of being aware of unknown speakers without compromising known speaker identification. To boost the overall performance of the PUKWS pipeline, we employ a cutting-edge data augmentation strategy that includes hard negative mining, rule-based procedures, GPT, and zero-shot voice cloning, thereby enhancing both M-UDKWS and SRPL components. Through exhaustive evaluations on various datasets and testing scenarios, we demonstrate the efficacy of our methods.

<p align="center">
  <img src="http://zhiyongchenGREAT.github.io/images/pukws.png" alt="pukws" width="50%" />
</p>

[Research paper: PUKWS](http://zhiyongchenGREAT.github.io/files/Personlized_UDKWS202401.pdf)

**MM-KWS: Multi-modal Prompts for Multilingual User-defined Keyword Spotting**

In this paper, we propose MM-KWS, a novel approach to user-defined keyword spotting leveraging multi-modal enrollments of text and speech templates. Unlike previous meth- ods that focus solely on either text or speech features, MM-KWS extracts phoneme, text, and speech embeddings from both modalities. These embeddings are then compared with the query speech embedding to detect the target keywords. To ensure the applicability of MM-KWS across diverse languages, we utilize a feature extractor incorporating several multilingual pre-trained models. Subsequently, we validate its effectiveness on Mandarin and English tasks. In addition, we have integrated advanced data augmentation tools for hard case mining to en- hance MM-KWS in distinguishing confusable words. Experimental results on the LibriPhrase and WenetPhrase datasets demonstrate that MM-KWS outperforms prior methods significantly.


<p align="center">
  <img src="http://zhiyongchenGREAT.github.io/images/mmkws.png" alt="mmkws" width="50%" />
</p>

[Research paper: MM-KWS](http://zhiyongchenGREAT.github.io/files/2406.07310v1.pdf)

[MM-KWS: Project Website](https://github.com/zhiyongchenGREAT/MM-KWS)

**SRPL: Open-set Speaker Identification with Reciprocal Points**

We introduce Personalized User-Defined Keyword Spotting (PUKWS), a novel pipeline specifically designed for enhancing household environments by integrating user-defined keyword spotting (KWS) with open-set speaker identification (SID) into a cascading dual sub-system structure. For KWS, we present multi-modal user-defined keyword spotting (M-UDKWS), a novel approach that leverages multi-modal prompts for text-audio multimodal enrollment, and optimizes phonetic and semantic feature extraction to synergize text and audio modalities. This innovation not only stabilizes detection by reducing mismatches between query audio and support text embeddings but also excels in handling potentially confusing keywords. For open-set SID, we adopt advanced open-set learning techniques to propose speaker reciprocal points learning (SRPL), addressing the significant challenge of being aware of unknown speakers without compromising known speaker identification. To boost the overall performance of the PUKWS pipeline, we employ a cutting-edge data augmentation strategy that includes hard negative mining, rule-based procedures, GPT, and zero-shot voice cloning, thereby enhancing both M-UDKWS and SRPL components. Through exhaustive evaluations on various datasets and testing scenarios, we demonstrate the efficacy of our methods.

<p align="center">
  <img src="http://zhiyongchenGREAT.github.io/images/srpl.png" alt="srpl" width="50%" />
</p>

[Research paper: Open-set Speaker Recognition](http://zhiyongchenGREAT.github.io/files/SRPL_slt24-5.pdf)

[SRPL: Project Website](https://srplplus.github.io) 

**Learning Domain-Heterogeneous Speaker Recognition Systems with Personalized Continual Federated Learning**

Speaker recognition, the process of automatically identifying a speaker based on individual characteristics in speech signals, presents significant challenges when addressing heterogeneous-domain conditions. Federated learning, a recent development in machine learning methods, has gained traction in privacy-sensitive tasks, such as personal voice assistants in home environments. However, its application in heterogeneous multi-domain scenarios for enhancing system customization remains underexplored. In this paper, we propose the utilization of federated learning in heterogeneous situations to enable adaptation across multiple domains. We also introduce a personalized federated learning algorithm designed to effectively leverage limited domain data, resulting in improved learning outcomes. Furthermore, we present a strategy for implementing the federated learning algorithm in practical, real-world continual learning scenarios, demonstrating promising results. The proposed federated learning method exhibits superior performance across a range of synthesized complex conditions and continual learning settings, compared to conventional training methods.

<p align="center">
  <img src="http://zhiyongchenGREAT.github.io/images/fedspk.png" alt="fedspk" width="50%" />
</p>

[Research paper: FedSpeaker](http://zhiyongchenGREAT.github.io/files/FedSpk.pdf) 

[FedSPK: Project Website](https://github.com/zhiyongchenGREAT/FedSPK) 

**Research on Domain Roubust Speaker Recognition**
Speaker recognition technology has advanced significantly, achieving high accuracy in controlled settings. However, in real-world applications, systems often face challenges due to domain variability—differences in recording environments, channels, languages, and demographic characteristics. Domain robust speaker recognition focuses on developing models that can maintain performance across these diverse conditions.

[Research paper: DA-Spk]([Supervised Imbalanced Multi-domain Adaptation for Text-independent Speaker Verification](https://www.researchgate.net/profile/Zhiyong-Chen-27/publication/348413798_Supervised_Imbalanced_Multi-domain_Adaptation_for_Text-independent_Speaker_Verification/links/621593d5ba15e05e2ea21019/Supervised-Imbalanced-Multi-domain-Adaptation-for-Text-independent-Speaker-Verification.pdf)) 

[Research paper: DA-Spk]([Supervised Imbalanced Multi-domain Adaptation for Text-independent Speaker Verification](https://www.researchgate.net/profile/Zhiyong-Chen-27/publication/348413798_Supervised_Imbalanced_Multi-domain_Adaptation_for_Text-independent_Speaker_Verification/links/621593d5ba15e05e2ea21019/Supervised-Imbalanced-Multi-domain-Adaptation-for-Text-independent-Speaker-Verification.pdf)) 

[Research paper: DA-Spk](https://www.researchgate.net/profile/Zhiyong-Chen-27/publication/348413798_Supervised_Imbalanced_Multi-domain_Adaptation_for_Text-independent_Speaker_Verification/links/621593d5ba15e05e2ea21019/Supervised-Imbalanced-Multi-domain-Adaptation-for-Text-independent-Speaker-Verification.pdf)

[Research paper: AM-Spk](https://arxiv.org/pdf/1908.03990)

[Research paper: Triplet-Spk](https://ieeexplore.ieee.org/abstract/document/9023253)

[Project Website and Code Repo](https://github.com/zhiyongchenGREAT/GREAT_ASV_system) 

Publications
------
*Learning domain-heterogeneous speaker recognition systems with personalized continual federated learning*, EURASIP Journal on Audio Speech and Music Processing. **Zhiyong Chen**, Shugong Xu

*Optimizing Feature Fusion for Improved Zero-shot Adaptation in Text-to-Speech Synthesis*, EURASIP Journal on Audio Speech and Music Processing. **Zhiyong Chen**, Zhiqi Ai, Xinnuo Li, Shugong Xu

*Enhancing Open-Set Speaker Identification through Rapid Tuning with Speaker Reciprocal Points and Negative Samples*, IEEE SLT2024. **Zhiyong Chen**, Zhiqi Ai, Xinnuo Li, Shugong Xu

*Personalized User-Defined Keyword Spotting in Household Environments: A Text-Audio Multi-Modality Approach*, Speech Communication. Zhiqi Ai, **Zhiyong Chen**, Xinnuo Li, Shugong Xu (In Peer Review)

*StyleFusion-TTS: Multimodal Style-control and Enhanced Feature Fusion for Zero-shot Text-to-speech Synthesis*, PRCV 2024. **Zhiyong Chen**, Zhiqi Ai, Xinnuo Li, Shugong Xu

*MM-KWS: Multi-modal Prompts for Multilingual User-defined Keyword Spotting*, Interspeech 2024. Zhiqi Ai, **Zhiyong Chen**, Xinnuo Li, Shugong Xu

*Supervised Imbalanced Multi-domain Adaptation for Text-independent Speaker Verification*, 2020 9th International Conference on Computing and Pattern Recognition. **Zhiyong Chen**, Zongze Ren, Shugong Xu

*ERDBF: Embedding-Regularized Double Branches Fusion for Multi-Modal Age Estimation*, IEEE Access. Bo Wu, Hengjie Lu, **Zhiyong Chen**, Shugong Xu

*Triplet Based Embedding Distance and Similarity Learning for Text-independent Speaker Verification* 2019 IEEE APSIPA ASC. Zongze Re, **Zhiyong Chen**, Shugong Xu

*A Study on Angular Based Embedding Learning for Text-independent Speaker Verification*, 2019 IEEE APSIPA ASC. **Zhiyong Chen**, Zongze Ren, Shugong Xu

*IFR: Iterative Fusion Based Recognizer for Low Quality Scene Text Recognition*, PRCV 2021. Zhiwei Jia, Shugong Xu, Shiyi Mu, **Zhiyong Chen**

*Configurable CNN Accelerator in Speech Processing based on Vector Convolution*, 2022 IEEE 4th International Conference on Artificial Intelligence Circuits and Systems (AICAS). Lanqing Hui, Shan Cao, **Zhiyong Chen**, Shugong Xu
<!-- Create content & metadata
------
For site content, there is one markdown file for each type of content, which are stored in directories like _publications, _talks, _posts, _teaching, or _pages. For example, each talk is a markdown file in the [_talks directory](https://github.com/academicpages/academicpages.github.io/tree/master/_talks). At the top of each markdown file is structured data in YAML about the talk, which the theme will parse to do lots of cool stuff. The same structured data about a talk is used to generate the list of talks on the [Talks page](https://academicpages.github.io/talks), each [individual page](https://academicpages.github.io/talks/2012-03-01-talk-1) for specific talks, the talks section for the [CV page](https://academicpages.github.io/cv), and the [map of places you've given a talk](https://academicpages.github.io/talkmap.html) (if you run this [python file](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.py) or [Jupyter notebook](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.ipynb), which creates the HTML for the map based on the contents of the _talks directory). -->

<!-- **Markdown generator** -->

<!-- I have also created [a set of Jupyter notebooks](https://github.com/academicpages/academicpages.github.io/tree/master/markdown_generator
) that converts a CSV containing structured data about talks or presentations into individual markdown files that will be properly formatted for the academicpages template. The sample CSVs in that directory are the ones I used to create my own personal website at stuartgeiger.com. My usual workflow is that I keep a spreadsheet of my publications and talks, then run the code in these notebooks to generate the markdown files, then commit and push them to the GitHub repository. -->

<!-- How to edit your site's GitHub repository
------
Many people use a git client to create files on their local computer and then push them to GitHub's servers. If you are not familiar with git, you can directly edit these configuration and markdown files directly in the github.com interface. Navigate to a file (like [this one](https://github.com/academicpages/academicpages.github.io/blob/master/_talks/2012-03-01-talk-1.md) and click the pencil icon in the top right of the content preview (to the right of the "Raw | Blame | History" buttons). You can delete a file by clicking the trashcan icon to the right of the pencil icon. You can also create new files or upload files by navigating to a directory and clicking the "Create new file" or "Upload files" buttons.  -->

<!-- Example: editing a markdown file for a talk
![Editing a markdown file for a talk](/images/editing-talk.png) -->

For more info
------
For further information about my research and collaborations, feel free to reach out via email at zhiyongchen@shu.edu.cn.
