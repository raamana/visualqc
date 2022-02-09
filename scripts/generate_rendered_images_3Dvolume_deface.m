# Script to generated fully rendered cross-sectional 2D images of a 3D mri volume
#
# Dependencies:
#    Viewer3D https://www.mathworks.com/matlabcentral/fileexchange/21993-viewer3d
#    ImResizeN https://www.mathworks.com/matlabcentral/fileexchange/64516-imresizen-resize-an-n-dimensional-array
#
# Acknowledgements: Athena Theyers developed the initial version with all essential commands

# MODIFY THIS -- path to the base directory where all images are stored
path='';


list = cellstr(ls(path));
I=regexp(list, regexptranslate('wildcard', '*.nii.gz'));

# find all non-empty images
ind = find(not(cellfun('isempty',I)));

% configuring various properties
% The thresholds in the AlphaTable variable in option_* variables below work for most scans typically,
%   but they may need to be adjusted depending on your dataset, as below:
%   converting more of the 0s to 1s will include more of the scan, 1s to 0s will remove more noise.
options_bw.AlphaTable=[0 0 0 1 1 1 1 1 1 1];
options_bw.RenderType='bw';

options_shaded.AlphaTable=[0 0 0 1 1 1 1 1 1 1];
options_shaded.ColorTable=[0.5 0.5 0.5;0.5 0.5 0.5;0.5 0.5 0.5; ...
                           0.5 0.5 0.5;0.5 0.5 0.5;0.5 0.5 0.5;0.5 0.5 0.5];
options_shaded.RenderType='shaded';
options_shaded.ShadingMaterial='dull';

% generating rendered image for each Nifti volume
for m = transpose(ind)
    if ~exist(strcat(path,'png\',strrep(list{m},'.nii.gz','.png')),'file')
        try
            nii=load_nii(strcat(path,list{m}),[],[],[],[],[],1);
        catch
            reslice_nii(strcat(path,list{m}),strcat(path,strrep(list{m},'.nii','a.nii')));
            nii=load_nii(strcat(path,strrep(list{m},'.nii','a.nii')),[],[],[],[],[],1);
        end

        % computing mean and SD of intensity to set extreme volumes to 0
		A=double(nii.img(:,:,:,1));
		a=mean(A(A~=0));
		b=std(A(A~=0));
		A(A>(a+3*b))=0;

		% if voxels not isometric, resize to 1mmx1mmx1mm
		if std(nii.hdr.dime.pixdim(2:4))~=0
			A=imresizen(A,[nii.hdr.dime.pixdim(2),nii.hdr.dime.pixdim(3),nii.hdr.dime.pixdim(4)]);
		end

		# resample the image to ensure max is < 400
		if max(nii.hdr.dime.dim(2:4))<400
			A=imresizen(A,400/max(size(A)));
		end

        % first angle
		I=rot90(render(permute(imgaussfilt3(A,1.25),[3,1,2]),options_shaded),2);
		J=rot90(render(permute(imgaussfilt3(A,2.5),[3,1,2]),options_bw),2);
		K=I+J.^0.33;
		K=K/max(K(:));
		K=K.^3;
		imwrite(K,strcat(path,'png\',strrep(list{m},'.nii.gz','.png')),'png');

		% different angles of rotation
		for n=[-45 45]
			B=imrotate3_fast(A,[0 0 n]);
			J=rot90(render(permute(imgaussfilt3(B,1  ),[3,1,2]),options_bw),2);
			I=rot90(render(permute(imgaussfilt3(B,1.5),[3,1,2]),options_shaded),2);
			K=I+J.^0.33;
			K=K/max(K(:));
			K=K.^3;
			imwrite(K,strcat(path,'png\',strrep(list{m},'.nii.gz',num2str(n)),'.png'));
		end
    end
end
