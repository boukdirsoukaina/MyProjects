Image = imread('image5.jpg');
Image1 = imread('image3.jpg');

face_Detector = vision.CascadeObjectDetector();

% localiser l'image
location_of_the_face1 = step(face_Detector, Image);
location_of_the_face2 = step(face_Detector, Image1);

detected_Image1 = insertShape(Image,'Rectangle', location_of_the_face1);
detected_Image2 = insertShape(Image1,'Rectangle', location_of_the_face2);

figure;
subplot(1,2,1) ;imshow(detected_Image1);title('Detection of One Face'); 
subplot(1,2,2) ;imshow(detected_Image2);title('Detection of Many Faces');