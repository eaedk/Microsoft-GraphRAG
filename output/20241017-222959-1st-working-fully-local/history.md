(.venv) emmanuelkoupoh@Emmanuels-MacBook-Air Microsoft-GraphRAG % python -m graphrag.index --root .

Logging enabled at 
/Users/emmanuelkoupoh/Documents/Github/Microsoft-GraphRAG/output/20241017-222959/reports/indexi
ng-engine.log
/Users/emmanuelkoupoh/Documents/Github/Microsoft-GraphRAG/.venv/lib/python3.11/site-packages/nu
mpy/core/fromnumeric.py:59: FutureWarning: 'DataFrame.swapaxes' is deprecated and will be 
removed in a future version. Please use 'DataFrame.transpose' instead.
  return bound(*args, **kwds)
🚀 create_base_text_units
                                 id  ... n_tokens
0  f2caae78925912c54b1f188557b5ab2c  ...      541

[1 rows x 5 columns]
🚀 create_base_extracted_entities
                                        entity_graph
0  <graphml xmlns="http://graphml.graphdrawing.or...
🚀 create_summarized_entities
                                        entity_graph
0  <graphml xmlns="http://graphml.graphdrawing.or...
🚀 create_base_entity_graph
   level                                    clustered_graph
0      0  <graphml xmlns="http://graphml.graphdrawing.or...
/Users/emmanuelkoupoh/Documents/Github/Microsoft-GraphRAG/.venv/lib/python3.11/site-packages/nu
mpy/core/fromnumeric.py:59: FutureWarning: 'DataFrame.swapaxes' is deprecated and will be 
removed in a future version. Please use 'DataFrame.transpose' instead.
  return bound(*args, **kwds)
/Users/emmanuelkoupoh/Documents/Github/Microsoft-GraphRAG/.venv/lib/python3.11/site-packages/nu
mpy/core/fromnumeric.py:59: FutureWarning: 'DataFrame.swapaxes' is deprecated and will be 
removed in a future version. Please use 'DataFrame.transpose' instead.
  return bound(*args, **kwds)
🚀 create_final_entities
                                  id  ...                              description_embedding
0   20de48aa312943c08278641be603b77f  ...  [0.015069414, 0.062229946, -0.17034593, -0.059...
1   47c07cf1b26e44ecb7e9ac6626b53691  ...  [0.038599897, 0.044208974, -0.16735856, -0.064...
2   3f6a7d3bca1c49cf8de7fe00b6127f9d  ...  [0.04396927, 0.015267911, -0.16092536, -0.0685...
3   da02d69cd2d24bc3a032e16369a77ea9  ...  [0.00375738, 0.028598137, -0.17081381, -0.0246...
4   ec554d0c538248a286c98130e2e5aa89  ...  [0.023604883, 0.02566571, -0.17407471, -0.0409...
5   e010471ee5974a49bc6bd8f32331e7cf  ...  [-0.012037484, 0.06705209, -0.17584185, -0.019...
6   2101253ff7d34f7c8be972ab8d8f7c1e  ...  [-0.0052363225, 0.10371297, -0.15878843, -0.03...
7   4d6982b39a2a42b29b2e0d53d76e4d64  ...  [0.035002094, 0.08719013, -0.13000137, -0.0221...
8   7c9c429551b04fbf806c478a57fa2557  ...  [0.018978372, 0.032936722, -0.17807, -0.007668...
9   48bb3e3788ee49feafca49f75e8e7d96  ...  [-0.07611797, 0.026442386, -0.18573822, -0.033...
10  8299cdde87ac4b70ba81adc13497ee01  ...  [0.039554413, 0.07472088, -0.14576659, -0.0474...
11  10030327c5f54081ad0f0cbc05c1d5db  ...  [0.04615785, 0.0129525075, -0.18815745, -0.046...
12  ef99476ea74646c586295fbecab74ec5  ...  [-0.007399403, 0.06414964, -0.16248299, -0.073...

[13 rows x 8 columns]
/Users/emmanuelkoupoh/Documents/Github/Microsoft-GraphRAG/.venv/lib/python3.11/site-packages/nu
mpy/core/fromnumeric.py:59: FutureWarning: 'DataFrame.swapaxes' is deprecated and will be 
removed in a future version. Please use 'DataFrame.transpose' instead.
  return bound(*args, **kwds)
/Users/emmanuelkoupoh/Documents/Github/Microsoft-GraphRAG/.venv/lib/python3.11/site-packages/da
tashaper/engine/verbs/convert.py:72: FutureWarning: errors='ignore' is deprecated and will 
raise in a future version. Use to_datetime without passing `errors` and catch exceptions 
explicitly instead
  datetime_column = pd.to_datetime(column, errors="ignore")
/Users/emmanuelkoupoh/Documents/Github/Microsoft-GraphRAG/.venv/lib/python3.11/site-packages/da
tashaper/engine/verbs/convert.py:72: UserWarning: Could not infer format, so each element will 
be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and 
as-expected, please specify a format.
  datetime_column = pd.to_datetime(column, errors="ignore")
🚀 create_final_nodes
    level                          title          type  ...                 top_level_node_id  
x  y
0       0  CONVOLUTIONAL NEURAL NETWORKS  ORGANIZATION  ...  20de48aa312943c08278641be603b77f  
0  0
1       0                   LECUN ET AL.        PERSON  ...  47c07cf1b26e44ecb7e9ac6626b53691  
0  0
2       0                          LENET                ...  3f6a7d3bca1c49cf8de7fe00b6127f9d  
0  0
3       0                           RELU  ORGANIZATION  ...  da02d69cd2d24bc3a032e16369a77ea9  
0  0
4       0                        DROPOUT  ORGANIZATION  ...  ec554d0c538248a286c98130e2e5aa89  
0  0
5       0                     MOBILENETS  ORGANIZATION  ...  e010471ee5974a49bc6bd8f32331e7cf  
0  0
6       0                  EFFICIENTNETS  ORGANIZATION  ...  2101253ff7d34f7c8be972ab8d8f7c1e  
0  0
7       0     NEURAL ARCHITECTURE SEARCH  ORGANIZATION  ...  4d6982b39a2a42b29b2e0d53d76e4d64  
0  0
8       0      RECURRENT NEURAL NETWORKS  ORGANIZATION  ...  7c9c429551b04fbf806c478a57fa2557  
0  0
9       0                   TRANSFORMERS  ORGANIZATION  ...  48bb3e3788ee49feafca49f75e8e7d96  
0  0
10      0                       GENOMICS  ORGANIZATION  ...  8299cdde87ac4b70ba81adc13497ee01  
0  0
11      0                CLIMATE SCIENCE  ORGANIZATION  ...  10030327c5f54081ad0f0cbc05c1d5db  
0  0
12      0                 ART GENERATION  ORGANIZATION  ...  ef99476ea74646c586295fbecab74ec5  
0  0

[13 rows x 14 columns]
🚀 create_final_communities
  id  ...                       text_unit_ids
0  0  ...  

[1 rows x 5 columns]
🚀 create_final_relationships
                           source                      target  ...  target_degree rank
0   CONVOLUTIONAL NEURAL NETWORKS                LECUN ET AL.  ...              1   13
1   CONVOLUTIONAL NEURAL NETWORKS                       LENET  ...              1   13
2   CONVOLUTIONAL NEURAL NETWORKS                        RELU  ...              1   13
3   CONVOLUTIONAL NEURAL NETWORKS                     DROPOUT  ...              1   13
4   CONVOLUTIONAL NEURAL NETWORKS                  MOBILENETS  ...              1   13
5   CONVOLUTIONAL NEURAL NETWORKS               EFFICIENTNETS  ...              1   13
6   CONVOLUTIONAL NEURAL NETWORKS  NEURAL ARCHITECTURE SEARCH  ...              1   13
7   CONVOLUTIONAL NEURAL NETWORKS   RECURRENT NEURAL NETWORKS  ...              1   13
8   CONVOLUTIONAL NEURAL NETWORKS                TRANSFORMERS  ...              1   13
9   CONVOLUTIONAL NEURAL NETWORKS                    GENOMICS  ...              1   13
10  CONVOLUTIONAL NEURAL NETWORKS             CLIMATE SCIENCE  ...              1   13
11  CONVOLUTIONAL NEURAL NETWORKS              ART GENERATION  ...              1   13

[12 rows x 10 columns]
🚀 create_final_text_units
                                 id  ...                                   relationship_ids
0  f2caae78925912c54b1f188557b5ab2c  ...  [b04a711442ad449d9602b58e86f3ecb3, c1c498ab3df...

[1 rows x 6 columns]
🚀 create_final_community_reports
  community  ...                                    id
0         0  ...  9054e929-473a-48a1-9c40-29f5b7060406

[1 rows x 10 columns]
/Users/emmanuelkoupoh/Documents/Github/Microsoft-GraphRAG/.venv/lib/python3.11/site-packages/da
tashaper/engine/verbs/convert.py:72: FutureWarning: errors='ignore' is deprecated and will 
raise in a future version. Use to_datetime without passing `errors` and catch exceptions 
explicitly instead
  datetime_column = pd.to_datetime(column, errors="ignore")
🚀 create_base_documents
                                 id  ...          title
0  9175421a52474b3266f1092e810ea5b8  ...  CNN_intro.txt

[1 rows x 4 columns]
🚀 create_final_documents
                                 id  ...          title
0  9175421a52474b3266f1092e810ea5b8  ...  CNN_intro.txt

[1 rows x 4 columns]
⠴ GraphRAG Indexer 
├── Loading Input (InputFileType.text) - 1 files loaded (0 filtered) ━━━━━━ 100% 0:00:… 0:00:00
├── create_base_text_units
├── create_base_extracted_entities
├── create_summarized_entities
├── create_base_entity_graph
├── create_final_entities
├── create_final_nodes
├── create_final_communities
├── create_final_relationships
├── create_final_text_units
├── create_final_community_reports
├── create_base_documents
└── create_final_documents
🚀 All workflows completed successfully.

---
(.venv) emmanuelkoupoh@Emmanuels-MacBook-Air Microsoft-GraphRAG % python -m graphrag.query --root ./ --method global "What is machine learning?"


creating llm client with {'api_key': 'REDACTED,len=6', 'type': "openai_chat", 'model': 'cas/ministral-8b-instruct-2410_q4km', 'max_tokens': 4000, 'temperature': 0.0, 'top_p': 1.0, 'n': 1, 'request_timeout': 600000.0, 'api_base': 'http://localhost:11434/v1', 'api_version': None, 'organization': None, 'proxy': None, 'cognitive_services_endpoint': None, 'deployment_name': None, 'model_supports_json': True, 'tokens_per_minute': 0, 'requests_per_minute': 0, 'max_retries': 10, 'max_retry_wait': 10.0, 'sleep_on_rate_limit_recommendation': True, 'concurrent_requests': 5}

SUCCESS: Global Search Response:
Machine Learning (ML) is a subset of artificial intelligence that involves training algorithms to make predictions or decisions based on data, without being explicitly programmed [Data: Relationships (0)]. It encompasses various techniques such as supervised learning, unsupervised learning, and reinforcement learning.

---
(.venv) emmanuelkoupoh@Emmanuels-MacBook-Air Microsoft-GraphRAG % python -m graphrag.query --root ./ --method local "What is machine learning?"



INFO: Vector Store Args: {}
[2024-10-17T22:46:43Z WARN  lance::dataset] No existing dataset at /Users/emmanuelkoupoh/Documents/Github/Microsoft-GraphRAG/output/20241017-222959/artifacts/lancedb/entity_description_embeddings.lance, it will be created
creating llm client with {'api_key': 'REDACTED,len=6', 'type': "openai_chat", 'model': 'cas/ministral-8b-instruct-2410_q4km', 'max_tokens': 4000, 'temperature': 0.0, 'top_p': 1.0, 'n': 1, 'request_timeout': 600000.0, 'api_base': 'http://localhost:11434/v1', 'api_version': None, 'organization': None, 'proxy': None, 'cognitive_services_endpoint': None, 'deployment_name': None, 'model_supports_json': True, 'tokens_per_minute': 0, 'requests_per_minute': 0, 'max_retries': 10, 'max_retry_wait': 10.0, 'sleep_on_rate_limit_recommendation': True, 'concurrent_requests': 5}
creating embedding llm client with {'api_key': 'REDACTED,len=6', 'type': "openai_embedding", 'model': 'nomic-embed-text', 'max_tokens': 4000, 'temperature': 0, 'top_p': 1, 'n': 1, 'request_timeout': 60000.0, 'api_base': 'http://localhost:11434/v1', 'api_version': None, 'organization': None, 'proxy': None, 'cognitive_services_endpoint': None, 'deployment_name': None, 'model_supports_json': None, 'tokens_per_minute': 0, 'requests_per_minute': 0, 'max_retries': 10, 'max_retry_wait': 10.0, 'sleep_on_rate_limit_recommendation': True, 'concurrent_requests': 5}

SUCCESS: Local Search Response:
Machine Learning (ML) is a subset of artificial intelligence that involves training algorithms to make predictions or decisions based on data, without being explicitly programmed. It leverages statistical techniques and computational methods to enable systems to learn from experience, improve performance over time, and adapt to new data.

### Key Components

1. **Supervised Learning**: This type of learning involves training a model on labeled data, where the correct answers are already known. The goal is for the model to learn patterns in the input data that can be used to predict outputs accurately.
2. **Unsupervised Learning**: In this approach, the algorithm learns from unlabeled data by identifying patterns and relationships within the dataset. It aims to discover hidden structures or groupings without predefined output variables.
3. **Reinforcement Learning**: This method involves an agent learning to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions, allowing it to learn optimal behavior over time.

### Applications

Machine Learning has a wide range of applications across various fields:

- **Healthcare**: Predictive analytics for disease diagnosis and treatment recommendations.
- **Finance**: Fraud detection, algorithmic trading, and credit scoring.
- **Retail**: Personalized product recommendations, inventory management, and customer segmentation.
- **Transportation**: Autonomous vehicles, route optimization, and traffic prediction.

### Techniques

Several techniques are commonly used in machine learning:

1. **Linear Regression**: Used for predicting a continuous output variable based on one or more input variables.
2. **Logistic Regression**: Used for binary classification problems where the goal is to predict the probability of an event occurring.
3. **Decision Trees and Random Forests**: These models use a tree-based approach to make decisions, with random forests combining multiple decision trees to improve accuracy.
4. **Support Vector Machines (SVM)**: Used for both classification and regression tasks by finding the optimal boundary or hyperplane that separates classes.
5. **Neural Networks**: Inspired by the human brain, neural networks consist of interconnected layers of nodes that process information and learn from data.

### Challenges

Despite its potential, machine learning faces several challenges:

- **Data Quality**: The performance of ML models heavily depends on the quality and quantity of training data.
- **Overfitting**: Models can become too complex and perform well on training data but poorly on new, unseen data.
- **Interpretability**: Some advanced models, like deep neural networks, are often considered "black boxes" due to their complexity.

### Future Directions

The future of machine learning is promising with ongoing research in areas such as:

- **Explainable AI (XAI)**: Developing methods to make ML models more interpretable.
- **Federated Learning**: Training models on decentralized data without exchanging it, ensuring privacy and security.
- **AutoML**: Automating the process of selecting and tuning machine learning algorithms for specific tasks.

In summary, machine learning is a powerful tool that has revolutionized various industries by enabling systems to learn from data and make intelligent decisions. Its applications are vast, ranging from healthcare to finance, transportation, and more. However, it also presents challenges that researchers continue to address through ongoing innovation and development.