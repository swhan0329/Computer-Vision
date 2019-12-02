function [H, rhoScale, thetaScale] = myHoughTransform(Im, threshold, rhoRes, thetaRes)
%Your implementation here
    Im_size = size(Im);
    limit =ceil(sqrt(Im_size(1)^2+Im_size(2)^2));
    
    rhoScale = -limit:rhoRes:limit;
    thetaScale = -pi:thetaRes:pi;
    thetaResd = thetaRes*(180/pi);
    
    H = zeros(size(rhoScale,2), size(thetaScale,2));
    
    %adjustment thr0eshold
    nonzeroV = nnz(Im);
    ratio = 0.3;
    thresholdV =0;
    while (ratio < 0.3 && ratio > 0.001) 
        thresholdV = 0;
        for i = 1:Im_size(1)
            for j = 1:Im_size(2)
                if Im(i,j)>threshold
                    thresholdV = thresholdV +1;
                end
            end
        end
        ratio = thresholdV/nonzeroV;
    
        if (ratio <  0.3)
            threshold = threshold - 0.0005;   
        else
            break;
        end
    end

    while (ratio < 1.0 && ratio > 0.7) 
        thresholdV=0;
        for i = 1:Im_size(1)
            for j = 1:Im_size(2)
                if Im(i,j)>threshold
                    thresholdV = thresholdV +1;
                end
            end
        end
    
        ratio = thresholdV/nonzeroV;
    
        if (ratio >  0.7)
            threshold = threshold + 0.001;     
        else
            break;
        end
    end
    
    
    %HoughTransform
    for i = 1:Im_size(1)
        for j = 1:Im_size(2)
            if Im(i,j)>threshold
                for theta = thetaScale
                    thetad = theta*(180/pi);
                    rho = j*cosd(thetad)+i*sind(thetad);
                    index_rho=floor((rho+limit)/rhoRes)+1;
                    index_theta=floor((thetad+180)/thetaResd)+1;
                    H(index_rho,index_theta) = H(index_rho,index_theta) + 1;
                end
            end
        end
    
    end
end        