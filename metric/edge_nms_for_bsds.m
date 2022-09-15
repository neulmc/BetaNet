clear; clc;
path_to_input = fullfile('./MAT_RESULTS_FOLDER/','tagfeeabs_redo-7');
path_to_output = fullfile('./NMS_RESULTS_FOLDER/','tagfeeabs_redo-7')

if ~exist(path_to_output,'dir')
    mkdir(path_to_output)
end

iids = dir(fullfile(path_to_input, '*.png'));
n=length(iids)
for i=1:n;
    edge = imread(fullfile(path_to_input, iids(i).name));
    dims = ndims(edge);
    if dims > 2
         edge = rgb2gray(edge); 
    end
    %edge = 1-single(edge)/255;
    edge = single(edge)/255;

    [Ox, Oy] = gradient2(convTri(edge, 4));
    [Oxx, ~] = gradient2(Ox);
    [Oxy, Oyy] = gradient2(Oy);
    O = mod(atan(Oyy .* sign(-Oxy) ./ (Oxx + 1e-5)), pi);
    % 2 for BSDS500 and Multi-cue datasets, 4 for NYUD dataset
    % NMS�����Խ����Ӱ�졣
    % ��function E = edgesNms(E0, O, r, s, m, nThreads)�У�
    % ��һ������E0��ʾ����ı�Ե��ǿ�ȣ����߽б�Ե���ʣ���
    % �ڶ�������O��ʾ��Ե�ķ���r��ʾNMS���õİ뾶��s��Ϊ������ͼ��߽總����������Ե��
    % s��ʾ������Ƶı߽��ȣ�m��ʾ�ѱ�Եǿ�ȳ���m��nThreads��ʾ���������߳̽���NMS��������Ҫ��Ҫ�����ǲ���r
    if (n == 200)
        edge = edgesNmsMex(edge, O, 2, 5, 1.01, 8); % from 1.01 to 1 # edgesNmsMex(edge, O, 2, 5, 1.01, 8)
        sss = iids(i).name;
        ppp = [sss(isstrprop([iids(i).name],'digit')) '.png'];
        imwrite(edge, fullfile(path_to_output, ppp));
    end
    if (n == 654)
        edge = edgesNmsMex(edge, O, 4, 5, 1.01, 8); % from 1.01 to 1 # edgesNmsMex(edge, O, 2, 5, 1.01, 8)
        sss = iids(i).name;
        ppp = sss;
        imwrite(edge, fullfile(path_to_output, ppp));
    end

    
end