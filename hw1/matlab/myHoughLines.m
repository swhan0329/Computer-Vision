function [rhos, thetas] = myHoughLines(H, nLines)
    %copy the rows, cols value of the img0
    [rows, cols]=size(H);
    
    %to make zero padding matrix
    padded_H = zeros(rows+2,cols+2);
    padded_H = cast(padded_H, class(padded_H));
    
    %put the original image to zero padding matirx 
    %It located in the middle of the zero padding matrix
    padded_H(2:end-1,2:end-1) = H;
    
    tempH = padded_H;
    padded_H_size = size(padded_H);
    
    for i = 2:padded_H_size(1)-1
        for j = 2:padded_H_size(2)-1
            t = padded_H(i,j);
            if padded_H(i-1,j-1)>=t || padded_H(i-1,j)>=t || padded_H(i-1,j+1)>=t 
                tempH(i,j) = 0;
            elseif  padded_H(i,j-1)>=t || padded_H(i,j+1)>=t               
                tempH(i,j) = 0;
            elseif padded_H(i+1,j-1)>=t || padded_H(i+1,j)>=t || padded_H(i+1,j+1)>=t
                tempH(i,j) = 0;
            end
        end
    end
    
    tempH = tempH(2:padded_H_size(1)-1,2:padded_H_size(2)-1);
    sorted_H = sort(tempH(:),'descend');
    
    sorted_H = sorted_H(1:nLines, 1);
    
    rhos = zeros(nLines,1);
    thetas = zeros(nLines,1);
    
    for num = 1:1:nLines
        [row_index,col_index] = find(tempH==sorted_H(num));
        r_size = size(row_index,1);
        c_size = size(col_index,1);
        tempH(row_index(floor(r_size/2)+1),col_index(floor(c_size/2)+1)) = 0;
        rhos(num,1) = row_index(floor(r_size/2)+1);
        thetas(num,1) = col_index(floor(c_size/2)+1);
    end
    
end
        