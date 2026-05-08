## Dataset attribution and ownership

This replication package uses third-party datasets for research and replication purposes. These datasets are not owned or authored by this project. Please cite and follow the terms of the original dataset sources.

### GitHub emotion polarity gold standard

This project uses the dataset **“A gold standard for polarity of emotions of software developers in GitHub”**, published on Figshare by **Nicole Novielli, Fabio Calefato, Davide Dongiovanni, Daniela Girardi, and Filippo Lanubile**.

Please cite the original dataset:

> Novielli, N., Calefato, F., Dongiovanni, D., Girardi, D., & Lanubile, F. (2020). *A gold standard for polarity of emotions of software developers in GitHub*. Figshare. https://doi.org/10.6084/m9.figshare.11604597.v1

### Pull request discussion sentiment dataset

This project also uses `dataset.json` from the replication package for **“Looks Good To Me ;-)”: Assessing Sentiment Analysis Tools for Pull Request Discussions**, provided by the `opus-research/sentiment-replication` repository.

According to that repository, `dataset.json` contains the dataset created for the study, including raw and preprocessed pull request discussion messages, GitHub message URLs, manually labeled polarity, expert-labeling confidence/agreement metadata, and tool predictions from sentiment analysis tools such as SentiStrength, SentiStrengthSE, SentiCR, DEVA, and Senti4SD.

Source repository:

> opus-research. *sentiment-replication: Replication package for “Looks Good To Me ;-)”: Assessing Sentiment Analysis Tools for Pull Request Discussions*. GitHub. https://github.com/opus-research/sentiment-replication

Dataset file:

> https://github.com/opus-research/sentiment-replication/blob/main/dataset.json

The dataset is not owned or authored by this project. It is used only for academic replication and evaluation.

### Russian GitHub dataset

This replication package includes `russian.csv`, a Russian-language GitHub pull request comment dataset compiled by **Mikalai Varapai** as part of this Bachelor End Project.

The original pull request comments were authored by GitHub users and are not authored or owned by this project. This project’s contribution consists of collecting publicly available GitHub PR comments, filtering them for Russian-language software engineering communication, deduplicating them, and organizing them into the `russian.csv` file for research and replication purposes.

Redistribution of this compiled CSV is permitted by the project author, provided that attribution to Mikalai Varapai is preserved. Users of the dataset remain responsible for respecting GitHub’s terms of service and any applicable rights or privacy considerations associated with the original comments.

Suggested attribution:

> Varapai, M. (2026). *Russian-language GitHub pull request comment dataset*. Compiled for the Bachelor End Project "Sentiment Analysis for Code Reviews", Eindhoven University of Technology.