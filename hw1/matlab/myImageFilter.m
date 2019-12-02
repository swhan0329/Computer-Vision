function [img1] = myImageFilter(img0, h)
    %copy the rows, cols value of the img0
    [rows, cols]=size(img0);
    [numRows, numCols]=size(h);
    
    %to make zero padding matrix
    padded_img = zeros(rows+2*floor(numRows/2),cols+2*floor(numRows/2));
    padded_img = cast(padded_img, class(padded_img));
    
    %put the original image to zero padding matirx 
    %It located in the middle of the zero padding matrix
    padded_img(floor(numRows/2)+1:end-floor(numRows/2),floor(numRows/2)+1:end-floor(numRows/2)) = img0;
    
    %result image initialize
    img1 = zeros(rows+2*floor(numRows/2),cols+2*floor(numRows/2));
    img1 = cast(img1, class(img0));
    
    %do convolution
    for i = floor(numRows/2)+1:1:rows+floor(numRows/2)-1
        for j = floor(numRows/2)+1:1:cols+floor(numRows/2)-1
            value = 0;
            for k = -floor(numRows/2):1:floor(numRows/2)
                for l = -floor(numRows/2):1:floor(numRows/2)
                    value = value + padded_img(i+k,j+l)*h(k+floor(numRows/2)+1,l+floor(numRows/2)+1);
                end
            end
            img1(i,j) = value;
        end
    end
    %crop the result image to input image size
    img1 = img1(floor(numRows/2)+1:end-floor(numRows/2),floor(numRows/2)+1:end-floor(numRows/2));
end
