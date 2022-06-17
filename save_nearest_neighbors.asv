% scenes = ["/Users/cameronfiore/C++/image_localization_project/data/chess/",...
%     "/Users/cameronfiore/C++/image_localization_project/data/fire/",...
%     "/Users/cameronfiore/C++/image_localization_project/data/heads/",...
%     "/Users/cameronfiore/C++/image_localization_project/data/office/",...
%     "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/",...
%     "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/",...
%     "/Users/cameronfiore/C++/image_localization_project/data/stairs/"];
clear;

scenes = ["/Users/cameronfiore/C++/image_localization_project/data/GreatCourt/",...
    "/Users/cameronfiore/C++/image_localization_project/data/KingsCollege/",...
    "/Users/cameronfiore/C++/image_localization_project/data/OldHospital/",...
    "/Users/cameronfiore/C++/image_localization_project/data/ShopFacade/",...
    "/Users/cameronfiore/C++/image_localization_project/data/StMarysChurch/",...
    "/Users/cameronfiore/C++/image_localization_project/data/Street/"];

setup;


for i = 1:length(scenes)
    scene = scenes(i);
    sequences = dir(scene);
    for j = 1:length(sequences)
        seq = sequences(j).name;
        pat = 'seq' + digitsPattern;
        if matches(seq, pat)
        if matches(seq, 'img')...
            || matches(seq, 'img_east')...
            || matches(seq, 'img_west')...
            || matches(seq, 'img_north')...
            || matches(seq, 'img_south')
            folder = append(scene,seq,'/');
            listing = dir(append(folder,'*.png'));
            imFns = cell(length(listing),1);
            for k=1:length(listing)
                imFns{k} = listing(k).name;
            end
            my_serial(folder, imFns);
        end
    end
end



DB_desc = zeros(4096, 8380);
DB_files = cell(8380, 1);

Q_desc = zeros(4096, 4841);
Q_files = cell(4841,1);

num1 = 1;
num2 = 1;

for i = 1:length(scenes)

    scene = scenes(i);
    sequences = dir(scene);

    train_images = [];
    test_images = [];

    for j = 1:length(sequences)
        seq = sequences(j).name;
        if seq == "dataset_train.txt"
            file = append(scene,seq);
            fileID = fopen(file, 'r');
            A = fscanf(fileID, '%s');
            fclose(fileID);

            B = regexp(A, 'seq\d\/frame\d\d\d\d\d.png', 'match');
            for b=1:length(B)
                train_images = [train_images, strcat(scene, B{b})];
            end

            C = regexp(A, 'img/image\d_\d\d\d\d\d\d.png', 'match');
            for b=1:length(C)
                train_images = [train_images, strcat(scene, C{b})];
            end

            D = regexp(A, 'img_east/image_east_\d\d\d\d.png', 'match');
            for b=1:length(D)
                train_images = [train_images, strcat(scene, D{b})];
            end

            E = regexp(A, 'img_north/image_north_\d_\d\d\d\d.png', 'match');
            for b=1:length(E)
                train_images = [train_images, strcat(scene, E{b})];
            end

            F = regexp(A, 'img_south/image_south_\d_\d\d\d\d.png', 'match');
            for b=1:length(F)
                train_images = [train_images, strcat(scene, F{b})];
            end

            G = regexp(A, 'img_west/image_west_\d\d\d\d.png', 'match');
            for b=1:length(G)
                train_images = [train_images, strcat(scene, G{b})];
            end

            H = regexp(A, 'seq\d\d\/frame\d\d\d\d\d.png', 'match');
            for b=1:length(H)
                train_images = [train_images, strcat(scene, H{b})];
            end

 
            
        elseif seq == "dataset_test.txt"
            file = append(scene,seq);
            fileID = fopen(file, 'r');
            A = fscanf(fileID, '%s');
            fclose(fileID);

            B = regexp(A, 'seq\d\/frame\d\d\d\d\d.png', 'match');
            for b=1:length(B)
                test_images = [test_images, strcat(scene, B{b})];
            end

            C = regexp(A, 'img/image\d_\d\d\d\d\d\d.png', 'match');
            for b=1:length(C)
                test_images = [test_images, strcat(scene, C{b})];
            end

            D = regexp(A, 'img_east/image_east_\d\d\d\d.png', 'match');
            for b=1:length(D)
                test_images = [test_images, strcat(scene, D{b})];
            end

            E = regexp(A, 'img_north/image_north_\d_\d\d\d\d.png', 'match');
            for b=1:length(E)
                test_images = [test_images, strcat(scene, E{b})];
            end

            F = regexp(A, 'img_south/image_south_\d_\d\d\d\d.png', 'match');
            for b=1:length(F)
                test_images = [test_images, strcat(scene, F{b})];
            end

            G = regexp(A, 'img_west/image_west_\d\d\d\d.png', 'match');
            for b=1:length(G)
                test_images = [test_images, strcat(scene, G{b})];
            end

            H = regexp(A, 'seq\d\d\/frame\d\d\d\d\d.png', 'match');
            for b=1:length(H)
                test_images = [test_images, strcat(scene, H{b})];
            end
        end
    end
    
    for f=1:length(train_images)
        file = train_images{f};
        load(append(file,'.mat'), 'd')
        DB_desc(:, num1) = d;
        DB_files{num1} = file;
        num1 = num1 + 1;
        if mod(f,100) == 0
            {'DB' num1-1}
        end
    end
    
    
    for f=1:length(test_images)
        file = test_images{f};
        load(append(file,'.mat'), 'd')
        Q_desc(:, num2) = d;
        Q_files{num2} = file;
        num2 = num2 + 1;
        if mod(f,100) == 0
            {'Q' num2-1}
        end
    end
    
end

[idx, dis] = yael_nn(DB_desc, Q_desc, 1000);

for i=1:length(Q_files)
    ids = idx(:, i);
    distances = dis(:, i);
    items = cell(length(ids), 2);
    for j=1:length(ids)
        items{j, 1} = DB_files{ids(j)};
        items{j, 2} = distances(j);
    end
    writecell(items,append(Q_files{i}, '.1000nn.txt'),'Delimiter','tab');
end