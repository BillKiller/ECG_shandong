for x in {0..4};
do
    python main.py train --kfold \_$x
done