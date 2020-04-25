# Semantic text similarity on short text segments

# Summary

Building on the [Semantic Text Similarity datasets](http://ixa2.si.ehu.es/stswiki/index.php/Main_Page) of SemEval,
this repository seeks to evaluate computationally-efficient approaches to identify short text segments
that have nearly the same semantic meaning in large-scale datasets.

# Getting started

The main entry point to the code in this repository is ``test_textsim.py``.
That file contains fuller comments describing the approaches being evaluated.

There are several similar files that build upon ``test_textsim.py``.
* ``test_unisent_multilingual.py`` evaluates the [Multilingual Universal Sentence Encoder](https://arxiv.org/abs/1907.04307) (which requires ``tensorflow>=2.0.0`` see comments in file)
* ``test_flair.py``evaluates various [transformer embeddings using the flairNLP library](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md)
* ``test_hindi.py`` uses XLM-R embeddings (from flairNLP) to test performance on a new Hindi dataset. That dataset is not in this repository currently. Please contact the maintainer (see below) if needed.

``density_plots.R`` plots out results, and ``mwe*`` files are minimal working examples for some approaches.

# Contact

Further information is available from Scott Hale. Meedan team members can 
contact Scott via Slack and others can reach out to Scott via comments/issues
on this repository or via [direct message on Twitter](https://twitter.com/computermacgyve)
