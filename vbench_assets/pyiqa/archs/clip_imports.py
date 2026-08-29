import warnings


with warnings.catch_warnings():
    warnings.filterwarnings(
        'ignore',
        message='pkg_resources is deprecated as an API.*',
        category=UserWarning,
    )
    import clip as clip
    from clip.simple_tokenizer import SimpleTokenizer


__all__ = ['clip', 'SimpleTokenizer']