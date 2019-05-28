function [ y ] = relog( x );
	y=x;
	y(y<0.)=0.;
	y=log(y+1.);
end

