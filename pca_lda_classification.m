% -----------------------------------------------------------------------------------------------
  
people = 40;
withinsample = 5;
principlenum = 50; % 預計降維到50維
target_dim =30; % 希望的LDA降維後維度
Row_FACE_Data = [];
training_data = [];
testing_data = [];
  
for k = 1:1:people
    for m = 1:2:10
        PathString = ['ORL3232', filesep, num2str(k), filesep, num2str(m), '.bmp'];
        ImageData = imread(PathString);
        ImageData = double(ImageData);
        if (k == 1 && m == 1)
            [row, col] = size(ImageData);
        end
        RowConcatenate = [];
        %--arrange the image into a vector
        for n = 1:row
            RowConcatenate = [RowConcatenate, ImageData(n, :)];
        end
        Row_FACE_Data = [Row_FACE_Data; RowConcatenate];
        training_data = [training_data; RowConcatenate];
    end
end
  
% class_labels = repelem(1:people, withinsample)';
class_labels = create_label(people, withinsample);

    
% zero mean 與 PCA 降維
MofData = mean(training_data);
NorData = training_data - MofData;
CMatrix = NorData' * NorData; 
[vec, val] = eig(CMatrix);
  
% 將特徵向量排序
val = diag(val);
[~, valindex] = sort(val);
vec = vec(:, valindex);
projectPCA = vec(:, 1:principlenum);
    
dimred = NorData * projectPCA;
    
% LDA 計算
num_classes = people;
num_features = size(dimred, 2);
    
% 確保目標維度不超過 LDA 的理論最大投影維度
tmp_array = [num_features, num_classes - 1];
max_lda_dim = findMin(tmp_array);
if target_dim > max_lda_dim
    error('目標維度 %d 超過 LDA 最大可用維度 %d', target_dim, max_lda_dim);
end
    
class_mean = zeros(num_classes, num_features);
for c = 1:num_classes
    class_mean(c, :) = mean(dimred(class_labels == c, :), 1);
end
    
% 計算類間散佈矩陣 (Sb) 和類內散佈矩陣 (Sw)
global_mean = mean(dimred, 1); 
Sb = zeros(num_features, num_features); 
Sw = zeros(num_features, num_features); 
    
for c = 1:num_classes
    Nc = sum(class_labels == c); 
    mean_diff = (class_mean(c, :) - global_mean)';
    Sb = Sb + Nc * (mean_diff * mean_diff'); 
    
    class_samples = dimred(class_labels == c, :);
    for i = 1:size(class_samples, 1)
        sample_diff = (class_samples(i, :) - class_mean(c, :))';
        Sw = Sw + (sample_diff * sample_diff');
    end
end
    
% 計算 Sw^-1 * Sb 並提取特徵值和特徵向量
[lda_vec, lda_val] = eig(pinv(Sw)*Sb );
%lda_val = diag(lda_val); % 提取特徵值
[~, lda_index] = sort(lda_val);
lda_vec = lda_vec(:, lda_index); 
    
lda_projection = lda_vec(:, 1:target_dim);
lda_result = dimred * lda_projection;
    
disp(size(lda_result));
  
  
% 測試
correct_predictions = 0;
for k = 1:1:people
      
    for m = 2:2:10
        PathString = ['ORL3232', filesep, num2str(k), filesep, num2str(m), '.bmp'];
        ImageData = imread(PathString);
        ImageData = double(ImageData);
        if (k == 1 && m == 1)
            [row, col] = size(ImageData);
        end
        RowConcatenate = [];
        %--arrange the image into a vector
        for n = 1:row
            RowConcatenate = [RowConcatenate, ImageData(n, :)];
        end
  
        % 對測試資料進行 zero mean 與 PCA 降維
        test_data = RowConcatenate;
        test_data = test_data - MofData;
        test_data = test_data * projectPCA; 
  
        test_result = test_data * lda_projection;
  
        distances = zeros(1, num_classes);
        for c = 1:num_classes
            class_mean_proj = class_mean(c, :) * lda_projection;
            distances(c) = calculate_distances(test_result, class_mean_proj);
        end
  
        [~, predicted_class] = findMin(distances);
  
        if predicted_class == k
            correct_predictions = correct_predictions + 1;
        end
    end
end
  
accuracy = correct_predictions / 200;
lda_result_3d = lda_result(:, 1:3);
figure;
cmap = hsv(40); 
scatter3(lda_result_3d(:, 1), lda_result_3d(:, 2), lda_result_3d(:, 3), 50, class_labels, 'filled');
  
title('3D LDA Projection');
xlabel('LDA Dimension 1');
ylabel('LDA Dimension 2');
zlabel('LDA Dimension 3');
  
colormap(cmap);
colorbar; 
  
grid on;
  
view(3); 
 
% -----------------------------------------------------------
function [minValue, minIndex] = findMin(array)
    minValue = array(1);
    minIndex = 1;
  
    for i = 2:length(array)
        if array(i) < minValue
            minValue = array(i);
            minIndex = i;
        end
    end
end
  
function distances = calculate_distances(test_result, class_mean_proj)
    % 計算測試資料與類別均值之間的距離，使用 D * D' 的概念
    %
    % test_result: 測試樣本的 LDA 投影結果 (1 × n_features)
    % class_mean_proj: 類別均值的 LDA 投影結果 (num_classes × n_features)
    %
    % 返回：
    % distances: 測試樣本與每個類別均值的距離 (1 × num_classes)
  
    % 計算差異矩陣 D (num_classes × n_features)
    D = class_mean_proj - test_result;
  

    distance_squared = sum(D .* D, 2);
      
    distances = sqrt(distance_squared);
end

function class_labels = create_label(people, withinsample)
    class_labels = [];
    
    total_samples = people * withinsample;
    
    for i = 1:total_samples
        label = ceil(i / withinsample);
        
        class_labels = [class_labels; label];
    end
end

function [sorted_array, sorted_index] = sort(input_array)
    % 自訂排序函數，模擬 MATLAB 的 sort 功能
    % input_array: 要排序的向量
    % sorted_array: 排序後的結果
    % sorted_index: 排序的索引
    
    n = length(input_array);
    
    sorted_array = input_array;
    sorted_index = 1:n;
    
    for i = 1:n-1
        for j = 1:n-i
            if sorted_array(j) < sorted_array(j+1)
                temp = sorted_array(j);
                sorted_array(j) = sorted_array(j+1);
                sorted_array(j+1) = temp;
                
                temp_index = sorted_index(j);
                sorted_index(j) = sorted_index(j+1);
                sorted_index(j+1) = temp_index;
            end
        end
    end
end


