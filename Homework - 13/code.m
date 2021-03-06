z1 = 0.2;
z2 = 0.2;
z3 = 0.2;
z4 = 0.2;
z5 = 0.2;
z6 = 0.2;

z7 = 0.3;
z8 = 0.3;
z9 = 0.3;
z10 = 0.3;
z11 = 0.3;
z12 = 0.3;

c1 = 0.7;
c2 = -0.3;

z1_new = round(0.7 * (0.2 + 0.2 + 0.2) - 0.3 * (0.2 + 0.2), 2);

z2_new = round(0.7 * (0.2 + 0.2 + 0.2 + 0.2) - 0.3 * (0.2 + 0.2), 2);

z3_new = round(0.7 * (0.2 + 0.2 + 0.2 + 0.2 + 0.2) - 0.3 * (0.2 + 0.3), 2);

z4_new = round(0.7 * (0.2 + 0.2 + 0.2 + 0.2 + 0.2) - 0.3 * (0.2 + 0.3 + 0.3) , 2);

z5_new = round(0.7 * (0.2 + 0.2 + 0.2 + 0.2 + 0.3) - 0.3 * (0.2 + 0.2 + 0.3 + 0.3), 2);

z6_new = round(0.7 * (0.2 + 0.2 + 0.2 + 0.3 + 0.3) - 0.3 * (0.2 + 0.2 + 0.3 + 0.3), 2);

z7_new = round(0.7 * (0.2 + 0.2 + 0.3 + 0.3 + 0.3) - 0.3 * (0.2 + 0.2 + 0.3 + 0.3), 2);

z8_new = round(0.7 * (0.2 + 0.3 + 0.3 + 0.3 + 0.3) - 0.3 * (0.2 + 0.2 + 0.3 + 0.3), 2);

z9_new = round(0.7 * (0.3 + 0.3 + 0.3 + 0.3 + 0.3) - 0.3 * (0.2 + 0.2 + 0.3 + 0.2), 2);

z10_new = round(0.7 * (0.3 + 0.3 + 0.3 + 0.3 + 0.3) - 0.3 * (0.2 + 0.3 + 0.2 + 0.2), 2);

z11_new = round(0.7 * (0.3 + 0.3 + 0.3 + 0.3 + 0.2) - 0.3 * (0.3 + 0.3 + 0.2 + 0.2), 2);

z12_new = round(0.7 * (0.3 + 0.3 + 0.3 + 0.2 + 0.2) - 0.3 * (0.3 + 0.3 + 0.2 + 0.2), 2);


fprintf('z1 new = %f\t\n', z1_new );
fprintf('z2 new = %f\t\n', z2_new );
fprintf('z3 new = %f\t\n', z3_new );
fprintf('z4 new = %f\t\n', z4_new );
fprintf('z5 new = %f\t\n', z5_new );
fprintf('z6 new = %f\t\n', z6_new );
fprintf('z7 new = %f\t\n', z7_new );
fprintf('z8 new = %f\t\n', z8_new );
fprintf('z9 new = %f\t\n', z9_new );
fprintf('z10 new = %f\t\n', z10_new );
fprintf('z11 new = %f\t\n', z11_new );
fprintf('z12 new = %f\t\n\n\n', z12_new );

Z_old = [z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12];

high_old = max(Z_old);
low_old = min(Z_old);
amplitude_old = high_old - low_old;

Z_new = [z1_new z2_new z3_new z4_new z5_new z6_new z7_new z8_new z9_new z10_new z11_new z12_new];

high_new = max(Z_new);
low_new = min(Z_new);
amplitude_new = high_new - low_new;

fprintf('Amplitude old = %f\t\n\n', amplitude_old );
fprintf('Amplitude new = %f\t\n', amplitude_new );

%plot(Z_old,'Linewidth',4);
%plot(Z_new,'Linewidth',4);

y = zeros(1,12);
y(1,9) = 0.78;
stem(Z_new,y,'Linewidth',4);
ylabel('Amplitude');
xlabel('Time Index');

