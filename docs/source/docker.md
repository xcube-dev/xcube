# Docker

To start a demo using docker use the following commands

    $ docker build -t [your name] .
    $ docker run -d -p [host port]:8000 [your name]
    
Example:

    $  docker build -t xcube:0.1.0dev6 .
    $  docker run -d -p 8001:8000 xcube:0.1.0dev6
    $  docker ps





    



