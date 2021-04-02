clear all

graphics_toolkit ("gnuplot");
pkg load signal;

global c_0=299792; % velocity of light in vacuum [km/s]
theta_e=12; % external beam angle from normal [°] at measurement

function data=spectrumread(file) % reads a calibrated OceanView spectrum, output is in wavelength [nm] and power ratio []
	data=csvread(file,"A14:A2061");
	data(:,2)=imag(data(:,1)./100); % power ratio converted from %
	data(:,1)=real(data(:,1)); % wavelength
end

function spectrumplot(data,range=[0,inf],ref=1,theta=0,plotname="") % plots a spectrum with wavelength axis from data in wavelength [nm] and power ratio []
	data(:,1)=data(:,1)/cos(deg2rad(theta)); % optional correction of spectral shift for tilted view of a Lippmann plate, where theta is internal light angle from normal [°]
	r=range(1)<=data(:,1)&data(:,1)<=range(2); % plot range (wavelength [nm])
	data(:,2)=data(:,2)./ref; % normalization to reference
	figure()
	hold on
	xlabel("\\lambda [nm]");
	ylabel("R []");
	plot(data(r,1),data(r,2));
	axis("tight","auto[y]")
	if !isempty(plotname)
		print(strcat("Plots/",plotname,".pdf"))
		csvwrite(strcat("Plotting_points/",plotname,".csv"),data(r,:))
	end
end

function spectrumplot_freq(data,range=[0,inf],ref=1,theta=0,spacing=0,plotname="") % plots a spectrum with frequency axis from data in wavelength [nm] and power ratio []
	global c_0;
	data(:,1)=data(:,1)/cos(deg2rad(theta)); % optional correction of spectral shift for tilted view of a Lippmann plate, where theta is internal light angle from normal [°]
	r=range(1)<=data(:,1)&data(:,1)<=range(2); % plot range (wavelength [nm])
	data(:,1)=c_0./data(:,1); % conversion to frequency [THz]
	data(:,2)=data(:,2)./ref; % normalization to reference
	figure()
	hold on
	xlabel("\\nu [THz]");
	ylabel("R []");
	plot(data(r,1),data(r,2));
	axis("tight","auto[y]")
	if spacing>0 % plot lines at multiple of the "spacing" frequency
		lines=spacing*ceil(min(data(r,1))/spacing):spacing:max(data(r,1));
		y_lim=get(gca,'ylim');
		for i=1:length(lines)
			plot([lines(i),lines(i)],y_lim,'k')
		end
	end
	if !isempty(plotname)
		print(strcat("Plots/",plotname,".pdf"))
		csvwrite(strcat("Plotting_points/",plotname,".csv"),data(r,:))
	end
end

% absolute calibration using mirror
Scal=spectrumread("4_Vacuum/mirror_Scalibration.txt"); % first reference (mirror compared to Spectralon, for S samples)
Lcal=spectrumread("4_Vacuum/mirror_Lcalibration.txt"); % second reference (mirror compared to Spectralon, for L samples)
points=Scal(:,1); % spectrometer points
Scal=Scal(:,2);
Lcal=Lcal(:,2);
mirror=dlmread("Silver_Coating_Comparsion_Data.csv",";","C4:D1981"); % mirror reflectance (Thorlabs P01 coating)
mirror(:,1)=mirror(:,1)*1000; % conversion from um to nm
mirror(:,2)=mirror(:,2)/100; % conversion from %
mirror=interp1(mirror(:,1),mirror(:,2),points); % interpolation at spectrometer points

% estimation of refractive index
spectrumplot(spectrumread("4_Vacuum/S2_unexposed2.txt"),[400,800],1./mirror,0,"S2_unexposed");
spectrumplot(spectrumread("4_Vacuum/L2_unexposed2.txt"),[400,800],1./mirror,0,"L2_unexposed");
R=0.042; % estimated reflectance [] (600 - 1000 nm)
n=(1+sqrt(R))/(1-sqrt(R)) % estimated refractive index [] (3rd order approximation in theta)
theta_i=rad2deg(asin(sin(deg2rad(theta_e))/n)) % internal beam angle from normal [°] at measurement

% relatively calibrated spectra (Spectralon reference, very noisy due to problem in OceanView)
spectrumplot(spectrumread("4_Vacuum/S2_marked.txt"),[0,inf],1,theta_i,"S2_Spectralon");
spectrumplot(spectrumread("4_Vacuum/L2_marked.txt"),[0,inf],1,theta_i,"L2_Spectralon");

% absolutely calibrated spectra
spectrumplot(spectrumread("4_Vacuum/S2_marked.txt"),[250,1000],10*Scal./mirror,theta_i,"S2");
spectrumplot(spectrumread("4_Vacuum/L2_marked.txt"),[250,1000],10*Lcal./mirror,theta_i,"L2");
delta_nu_S2=21.6;
spectrumplot_freq(spectrumread("4_Vacuum/S2_marked.txt"),[500,1000],10*Scal./mirror,theta_i,delta_nu_S2,"S2_fit");
delta_nu_L2=15.5;
spectrumplot_freq(spectrumread("4_Vacuum/L2_marked.txt"),[500,1000],10*Lcal./mirror,theta_i,delta_nu_L2,"L2_fit");

% estimated depth [nm]
Z_S2=c_0/(2*n*delta_nu_S2)
Z_L2=c_0/(2*n*delta_nu_L2)

pause
