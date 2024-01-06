function convertDicomToPng(folderPath)
    % 检查并处理当前文件夹中的所有 DICOM 文件
    files = dir(fullfile(folderPath, '*.dcm'));
    for i = 1:length(files)
        filePath = fullfile(files(i).folder, files(i).name);
        imageData = dicomread(filePath);
        
        % 选择序列中的第一个时间点，如果数据是四维的
        if ndims(imageData) == 4
            imageData = imageData(:,:,:,1);
        end
        
        % 将16位图像数据标准化到0-1范围
        imageData = double(imageData);
        imageData = (imageData - min(imageData(:))) / (max(imageData(:)) - min(imageData(:)));
        
        % 转换为8位数据
        imageData = uint8(255 * imageData);
        
        % 转换为 PNG 并保存
        pngFilePath = strrep(filePath, '.dcm', '.png');
        imwrite(imageData, pngFilePath);
    end
    
    % 递归处理所有子文件夹
    subFolders = dir(folderPath);
    subFolders = subFolders([subFolders.isdir]);
    subFolders = subFolders(~ismember({subFolders.name}, {'.', '..'}));
    
    for i = 1:length(subFolders)
        subFolderPath = fullfile(folderPath, subFolders(i).name);
        convertDicomToPng(subFolderPath);
    end
end
