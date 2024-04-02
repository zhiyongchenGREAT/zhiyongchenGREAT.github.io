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
As a current third year PhD student, I have the honor of being guided by Professor Shugong Xu, a distinguished IEEE Fellow (https://www.researchgate.net/profile/Shugong-Xu-2). My research is deeply rooted in the field of speech technology, encompassing speaker recognition, keyword recognition, speech synthesis, zero-shot voice cloning, federated learning, transfer learning, and few-shot learning. Presently, my active research interests include exploring audio large models, speech content synthesis, and multi-modal learning. I possess a profound interest in the applications and development trends of artificial intelligence technologies within the speech and audio domain.

My Active Research Projects
--------
**Optimizing Feature Fusion for Improved
Zero-shot Adaptation in Text-to-Speech
Synthesis**

A primary challenge in VC is maintaining speech quality and speaker similarity with limited reference data for a specific speaker. However, existing VC systems often rely on naive combinations of embedded speaker vectors for speaker control, which compromises the capture of speaking style, voice print, and semantic accuracy. To overcome this, we introduce the Two-branch Speaker Control Module (TSCM), an novel and highly adaptable voice cloning module designed to precisely processing speaker or style control for a target speaker. Our method uses an advanced fusion of local-level features from a Gated Convolutional Network (GCN) and utterance-level features from a Gated Recurrent Unit (GRU) to enhance speaker control. We demonstrate the effectiveness of TSCM by integrating it into advanced TTS systems like FastSpeech 2 and VITS architectures, significantly optimizing their performance. Experimental results show that TSCM enables accurate voice cloning for a target speaker with minimal data through both zero-shot or few-shot fine-tuning of pre-trained TTS models. Furthermore, our TSCM based VITS (TSCM-VITS) showcases superior performance in zero-shot scenarios compared to existing state-of-the-art VC systems, even with basic dataset configurations. Our method's superiority is validated through comprehensive subjective and objective evaluations.

[demo](https://great-research.github.io/tsct-tts-demo/) 

[paper](https://github.com/zhiyongchenGREAT/zhiyongchenGREAT.github.io/blob/master/files/Efficient_Speaker_Feature_Fusion_Module_for_Few_Shot_Voice_Cloning_System__EURASIP_version_submit2_.pdf)

**Personalized User-Defined Keyword Spotting and Open-set Speaker Identification in Household Environments**

We introduce Personalized User-Defined Keyword Spotting (PUKWS), a novel pipeline specifically designed for enhancing household environments by integrating user-defined keyword spotting (KWS) with open-set speaker identification (SID) into a cascading dual sub-system structure. For KWS, we present multi-modal user-defined keyword spotting (M-UDKWS), a novel approach that leverages multi-modal prompts for text-audio multimodal enrollment, and optimizes phonetic and semantic feature extraction to synergize text and audio modalities. This innovation not only stabilizes detection by reducing mismatches between query audio and support text embeddings but also excels in handling potentially confusing keywords. For open-set SID, we adopt advanced open-set learning techniques to propose speaker reciprocal points learning (SRPL), addressing the significant challenge of being aware of unknown speakers without compromising known speaker identification. To boost the overall performance of the PUKWS pipeline, we employ a cutting-edge data augmentation strategy that includes hard negative mining, rule-based procedures, GPT, and zero-shot voice cloning, thereby enhancing both M-UDKWS and SRPL components. Through exhaustive evaluations on various datasets and testing scenarios, we demonstrate the efficacy of our methods.

[paper1](https://github.com/zhiyongchenGREAT/zhiyongchenGREAT.github.io/blob/master/files/Personlized_UDKWS202401.pdf)

[paper2](https://github.com/zhiyongchenGREAT/zhiyongchenGREAT.github.io/blob/master/files/SRPL_IS24.pdf)

[project web](https://srplplus.github.io) 


**Learning Domain-Heterogeneous Speaker Recognition Systems with Personalized Continual Federated Learning**

Speaker recognition, the process of automatically identifying a speaker based on individual characteristics in speech signals, presents significant challenges when addressing heterogeneous-domain conditions. Federated learning, a recent development in machine learning methods, has gained traction in privacy-sensitive tasks, such as personal voice assistants in home environments. However, its application in heterogeneous multi-domain scenarios for enhancing system customization remains underexplored. In this paper, we propose the utilization of federated learning in heterogeneous situations to enable adaptation across multiple domains. We also introduce a personalized federated learning algorithm designed to effectively leverage limited domain data, resulting in improved learning outcomes. Furthermore, we present a strategy for implementing the federated learning algorithm in practical, real-world continual learning scenarios, demonstrating promising results. The proposed federated learning method exhibits superior performance across a range of synthesized complex conditions and continual learning settings, compared to conventional training methods.

[Paper](https://github.com/zhiyongchenGREAT/zhiyongchenGREAT.github.io/blob/master/files/Learning_Domain_Heterogeneous_Speaker_Recognition_Systems_with_Personalized_Continual_Federated_Learning__submission_R2_.pdf) 

[Opensource Code](https://github.com/zhiyongchenGREAT/FedSPK) 

Publications
------
*Learning domain-heterogeneous speaker recognition systems with personalized continual federated learning*, EURASIP Journal on Audio Speech and Music Processing. **Zhiyong Chen**, Shugong Xu

*Optimizing Feature Fusion for Improved Zero-shot Adaptation in Text-to-Speech Synthesis*, EURASIP Journal on Audio Speech and Music Processing. **Zhiyong Chen**, Zhiqi Ai, Xinnuo Li, Shugong Xu (Publishing)

*Enhancing Open-Set Speaker Identification through Rapid Tuning with Speaker Reciprocal Points and Negative Samples*, Interspeech24. **Zhiyong Chen**, Zhiqi Ai, Xinnuo Li, Shugong Xu (Just submitted)

*Personalized User-Defined Keyword Spotting in Household Environments: A Text-Audio Multi-Modality Approach*, Speech Communication. Zhiqi Ai, **Zhiyong Chen**, Xinnuo Li, Shugong Xu (Just submitted)

*Optimizing Feature Fusion for Improved Zero-shot Adaptation in Text-to-Speech Synthesis*, EURASIP Journal on Audio Speech and Music Processing. Zhiqi Ai, **Zhiyong Chen**, Xinnuo Li, Shugong Xu

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
