function [img1] = myEdgeFilter(img0, sigma)
    hsize =2*ceil(3*sigma)+1;
    G = fspecial('gaussian',hsize,sigma);
    img0 = myImageFilter(img0, G);
    
    xSobel = [1 0 -1;2 0 -2;1 0 -1]/8;
    ySobel = [1 2 1;0 0 0;-1 -2 -1]/8;

    Xgradient = myImageFilter(img0, xSobel);
    Ygradient = myImageFilter(img0, ySobel);
    
    Im = sqrt(Xgradient.^2 + Ygradient.^2);
    arctan = atan2(Ygradient,Xgradient);
    
    Temp = Im;
    Im_size = size(Im);
    
    %padding zero in order not to judge the edge of the pictuer itself as
    %an edge
    for i = 1:Im_size(1)
        for j = 1:Im_size(2) 
            Temp(i,1) = 0;
            Temp(i,2) = 0;
            Temp(1,j) = 0;
            Temp(2,j) = 0;
            Temp(i,Im_size(2)-1) = 0;
            Temp(i,Im_size(2)) = 0;
            Temp(Im_size(1)-1,j) = 0;
            Temp(Im_size(1),j) = 0;
        end
    end
    
    img1 = myNonMaximumSup(Im, Temp, arctan);
 
end
    
                
        
        
