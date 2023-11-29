# Sensitive Data Anonymization
Short coding challenge on anonymizing sensitive data.

## Installation

This application is written in [Python 3.10](https://www.python.org/downloads/) and depends on the following packages:
 - [fastAPI](https://fastapi.tiangolo.com/)
 - [pydantic](https://pydantic.dev/)
 - [spaCy](https://spacy.io/)

Once Python is setup, run the following command in your terminal of preference to install the required dependencies.
```
pip install -r requirements.txt
```

## Deployment
Once your Python environment is setup and you've downloaded the source code, the service can be easily run from the command line with the following command.
```
python <path_to_project>/main.py
```

By default, the REST API is exposed on `http://0.0.0.0:8888`. The host and port can respectively configured by setting the `APP_HOST` and `APP_PORT` environment variables.
