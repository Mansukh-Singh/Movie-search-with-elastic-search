indexMapping = {
    "properties":{
        "title":{
            "type":"keyword"
        },
        "rating":{
            "type":"float"
        },
        "genre":{
            "type":"nested",
            "properties":{
                "name":{
                    "type":"keyword"
                }
            }
        },
        "overview":{
            "type":"text"
        },
        "tagline":{
            "type":"text"
        },
        "languages":{
            "type":"nested",
            "properties":{
                "name":{
                    "type":"keyword"
                }
            }
        },
        "embeddings":{
            "type":"dense_vector",
            "dims":768,
            "similarity": "cosine"
        }
    }
}
