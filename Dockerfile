FROM python:3.7
EXPOSE 8344
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD streamlit run app.py --server.port 8344
