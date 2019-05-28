function [ y ] = logan( x );
	y=sign(x).*log(abs(x)+1);
end

