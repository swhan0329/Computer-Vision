function [img1] = myNonMaximumSup(Im, Temp, arctan)
    
    Im_size = size(Im);
    
    %Non Maximum Suppression
    for i = 2:Im_size(1)-2
        for j = 2:Im_size(2)-2
            
            %radian to degree
            x = arctan(i,j)*180/pi;
            if (180 <= x && x <360)
                x = x - 180;
            end
            
            y = Im(i,j);
            if (x < 22.5) || (157.5<=x) %0~22.5 or 157.5~180
                if (y<Im(i,j-1)) || (y<Im(i,j+1))
                    Temp(i,j) = 0;
                end
            elseif 22.5 <= x && x < 67.5 %22.5 ~ 67.5
                if (y<Im(i-1,j-1)) || (y<Im(i+1,j+1))
                    Temp(i,j) = 0;
                end
            elseif 67.5 <= x && x < 112.5 %67.5 ~ 112.5
                if (y<Im(i-1,j)) || (y<Im(i+1,j))
                    Temp(i,j) = 0;
                end
            elseif 112.5 <= x && x < 157.5 %112.5 ~ 157.5
                if (y<Im(i+1,j-1)) || (y<Im(i-1,j+1))
                    Temp(i,j) = 0;
                end
            end
        end
    end
    
    img1 = Temp;
end

