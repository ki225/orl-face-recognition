people = 40;  % 設定總人數為40
withinsample = 5;  % 每個人有5張樣本
Row_FACE_Data = [];  % 用來儲存所有圖像的矩陣
principlenum = 65;  % 設定PCA中使用的主成分數量
 
% Step 1: Load and preprocess images for PCA
for k = 1:people
    for m = 1:2:10 
        PathString = ['ORL3232', filesep, num2str(k), filesep, num2str(m), '.bmp'];
        ImageData = imread(PathString);
        ImageData = double(ImageData);
         
        RowConcatenate = reshape(ImageData, 1, []); 
        Row_FACE_Data = [Row_FACE_Data; RowConcatenate];
    end
end
 
% PCA
meanD = mean(Row_FACE_Data);
NorData = Row_FACE_Data - meanD;
CMatrix = NorData' * NorData;
[vec, val] = eig(CMatrix);
val = diag(val);
[junk, index] = sort(val, 'descend');
vec = vec(:, index);
projectPCA = vec(:, 1:principlenum);
pcaTotalFACE = NorData * projectPCA;
 
for i = 1:principlenum
    maxAttr = max(pcaTotalFACE(:, i));
    minAttr = min(pcaTotalFACE(:, i));
    pcaTotalFACE(:, i) = (pcaTotalFACE(:, i) - minAttr) / (maxAttr - minAttr);
end
 
% Step 2: Train neural network using PCA features
n_hidden = 150;
n_epoch = 300;
learning_rate = 0.5;
correct = 0;
 
outputmatrix = 2 * rand(n_hidden, 40) - 1;
hiddenmatrix = 2 * rand(principlenum, n_hidden) - 1;
 
target = zeros(200, 40);
for i = 1:200
    target(i, ceil(i / 5)) = 1;
end
 
avgloss = zeros(n_epoch, 1);
for epoch = 1:n_epoch
    epoch_loss = 0; 
     
    for i = 1:200
        trainingClass = ceil(i / 5);
        % Forward pass
        SUMhid = pcaTotalFACE(i, :) * hiddenmatrix;
        Ahid = logsig(SUMhid);
        SUMout = Ahid * outputmatrix;
        Aout = softmax(SUMout')';
         
        loss = -log(Aout(trainingClass));
        epoch_loss = epoch_loss + loss;
 
        % Backward pass
        DELTAout = target(i, :) - Aout;
        DELTAhid = DELTAout * outputmatrix' .* dlogsig(SUMhid, Ahid);
        outputmatrix = outputmatrix + learning_rate * Ahid' * DELTAout;
        hiddenmatrix = hiddenmatrix + learning_rate * pcaTotalFACE(i, :)' * DELTAhid;
    end
     
    avg_epoch_loss = epoch_loss / 200;
    if avg_epoch_loss > 0.5
        avgloss(epoch) = 0.5;
    else
        avgloss(epoch) = avg_epoch_loss;
    end
     
    fprintf('Epoch %d: avg_loss = %.4f\n', epoch, avgloss(epoch));
end
 
final_loss = avgloss(n_epoch);
fprintf('Final loss after %d epochs: %.4f\n', n_epoch, final_loss);
 
figure;
plot(1:n_epoch, avgloss(1:n_epoch));
legend("Training");
ylabel('avg_loss');
xlabel("Epoch");
 
% Step 3: Test the neural network with test data
Row_TEST_Data = [];
for k = 1:people
    for m = 2:2:10  % 只使用偶數編號的圖像（測試集）
        PathString = ['ORL3232', filesep, num2str(k), filesep, num2str(m), '.bmp'];
        ImageData = imread(PathString);
        ImageData = double(ImageData);
         
        RowConcatenate = reshape(ImageData, 1, []);
        Row_TEST_Data = [Row_TEST_Data; RowConcatenate];
    end
end
 
% Normalize test data
meanTest = mean(Row_TEST_Data);
NorTestData = Row_TEST_Data - meanTest;
pcaTestFACE = NorTestData * projectPCA;
 
% Normalize PCA test data
for i = 1:principlenum
    maxAttr = max(pcaTestFACE(:, i));
    minAttr = min(pcaTestFACE(:, i));
    pcaTestFACE(:, i) = (pcaTestFACE(:, i) - minAttr) / (maxAttr - minAttr);
end
 
for i = 1:200
    test_val = softmax((logsig(pcaTestFACE(i, :) * hiddenmatrix) * outputmatrix)')';
    [max_val, max_index] = max(test_val);
    if (max_index == ceil(i / 5))
        correct = correct + 1;
    end
end
 
accuracy = correct / 200;
fprintf("correct_percent = %.3f\n", accuracy);
