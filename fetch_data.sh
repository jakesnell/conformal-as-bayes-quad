if [ ! -d "data" ]; then
    echo "preparing data directory..."
    gdown 1h7S6N_Rx7gdfO3ZunzErZy6H7620EbZK -O data.tar.gz
    tar -xf data.tar.gz -C .
    rm data.tar.gz
else
    echo "data directory already exists"
fi
