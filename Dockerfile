FROM python:3.7
EXPOSE 3344
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD streamlit run app.py --server.port 3344
