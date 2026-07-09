from application.reranker.reranker import ReRanker


chunks = [

    {
        "text":
        "Azure Stack HCI upgrade requires environment validation"
    },

    {
        "text":
        "This document explains storage configuration"
    }

]


reranker = ReRanker()


result = reranker.rerank(

    "How to validate Azure Stack HCI upgrade?",

    chunks,

    1

)


print(result)