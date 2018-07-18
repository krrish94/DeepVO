data=importdata('01.txt');
fID = fopen('axisData.txt','w');

for i=1:size(data,1)
    row=reshape(data(i,:),4,3)';
    R=row(1:3,1:3);
    ax_ = rotm2axang(R);
    ax = ax_(1,1:3)*ax_(4);
    fprintf(fID, '%0.3f %0.3f %0.3f\n', ax(1,1), ax(1,2), ax(1,3));
    
end

fclose(fID);
    
    