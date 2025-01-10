## Qdrant experiment

To try out whether a single collection with a "user_id" field is faster than separate collections per user.


## Results

| Configuration | Search Time (ms) | Standard Deviation (ms) |
|--------------|------------------|------------------------|
| Separate Collections | 3.8 | 4.3 |
| Single Collection (no index) | 35.8 | 4.8 |
| Single Collection (keyword index) | 31.6 | 8.2 |

## Running the experiment

```
$ docker run -p 6333:6333 qdrant/qdrant
$ pip install requirements.txt
$ python claude_compare.py
```

## Conclusion

Separate collections are faster than a single collection.