D = '/home/scat5837/Documents/covid_test/v7/folders/covid';
S = dir(fullfile(D,'*'));
N = setdiff({S([S.isdir]).name},{'.','..'});

ref= imread('/home/scat5837/Documents/Atten_Deep_MIL/data/data/1/ffe6a2e9340dcf3e55b9619b4c1182e5/ffe6a2e9340dcf3e55b9619b4c1182e5_0113_p0.279_m21.385_c40496.png');

for ii = 1:numel(N)
    T = dir(fullfile(D,N{ii},'*.png')); % improve by specifying the file extension.
    C = {T(~[T.isdir]).name}; % files in subfolder.
    folder_name= N(ii);
    mkdir(folder_name{1});
    for jj = 1:numel(C)
        F = fullfile(D,N{ii},C{jj})
        split_fil = split(F,"/")
        f = fullfile(folder_name{1}, '/', split_fil{10})
        origin = imread(F);
        new= imhistmatch(origin, ref);
        imwrite(new, f);
        
        % do whatever with file F.
        
    end
end