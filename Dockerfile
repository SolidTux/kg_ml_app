FROM python:3.7
EXPOSE 8344
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN tar -jxf kostunin_trees.tar.bz2 && rm kostunin_trees.tar.bz2
CMD streamlit run app.py --server.port 8344
