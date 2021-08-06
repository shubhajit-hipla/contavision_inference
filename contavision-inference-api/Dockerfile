FROM ubuntu:18.04


RUN apt-get -y update && \
  apt-get install -y software-properties-common 

RUN apt-get install -y python3-pip \
    && apt-get clean \
    && python3 -m pip install --upgrade pip  


# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel


RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip3 install -r requirements.txt
COPY . /opt/app

#WORKDIR /app
#ADD . /app

#RUN pip3 install -r requirements.txt

#ADD requirements.txt /
#COPY . /app
##ADD utils /utils 
##COPY res/ /res

RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8

ENTRYPOINT [ "python3" ]


#CMD [ "webstreaming.py","--ip","0.0.0.0","--port","5000" ]
CMD [ "single_attendance_runner.py","6037800305e4a8f00accd473","SI31793","all" ]
#CMD [ "single_attendance_runner.py"] 
#CMD [ "app.py","--ip","0.0.0.0","--port","5000" ]



#ADD my_script.py /
#ADD run.sh /

#RUN chmod  777 /run.sh

# CMD [ "python3", "./my_script.py" ]
#CMD ["/bin/bash", "-c", "/run.sh"]

