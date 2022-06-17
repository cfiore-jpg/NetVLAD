# TODO before running

* Download model weights from https://www.di.ens.fr/willow/research/netvlad/ and store in "models" folder
* Format all database images into a cell called imFns and give as input to my_serial along with folder

imFns = {image_1.png, image_2.png, ..., image_N.png};

folder = \Users\joe_smith\KingsCollege\seq1\;

my_serial(folder, imFns);

* Once the descriptors are generated, format and generate DB_desc, DB_files, Q_desc, and Q_files appropriatley.

DB_desc = zeros(4096, num_database_images);

DB_files = cell(num_database_images, 1);

Q_desc = zeros(4096, num_query_images);
Q_files = cell(num_query_images, 1);

* Edit the number of nearest neighbors for the search in yael_nn(DB_desc, Q_desc, num_neighbors)
* PS: if you are just timing yael_nn, then ignore everything after it
