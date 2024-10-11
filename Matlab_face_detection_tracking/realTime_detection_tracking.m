clear all ;

cam = webcam();
%taille choisi
cam.Resolution = '424x240';

%video_frame = snapshot(cam); %Acquire One Image Frame from Webcam

video_player = vision.VideoPlayer('Position',[100 100 424 240]); %affichage des images vidéo

% the object that will be used to detect faces
face_detector = vision.CascadeObjectDetector(); %detects objects in images by sliding a window over the image
point_tracker = vision.PointTracker('MaxBidirectionalError',2);

runing_loop = true;
nb_of_points =0 ;
frame_nb = 0;

while runing_loop && frame_nb < 400
    
    video_frame = snapshot(cam);
    gray_frame = rgb2gray(video_frame);
    frame_nb = frame_nb + 1 ;
    
    if nb_of_points < 10
        face_rectangle = face_detector.step(gray_frame); % nous localisons le rectangle qui entoure le visage
        
        %if the face_rectangle is locolized
        if ~isempty(face_rectangle)
            %Detect interest points in the object region. object here is the face 
            % (1, :) just for one face.
            points = detectMinEigenFeatures(gray_frame,'ROI', face_rectangle(1, :) ); %roi = region of interest
            
            xy_points = points.Location;
            nb_of_points = size(xy_points, 1); % 1 means x to get
            
            release(point_tracker); %empty the point tracker
            initialize(point_tracker, xy_points, gray_frame); %initialize point tracker with xy_points on gray_frame
            
            previous_points = xy_points; %for later, we gonna compare the distance between these 2.
            
            rectangle = bbox2points(face_rectangle(1, :) ); %converting the rectangle,which is arround the face, into points
            
            %when face will move at different angles, the face rectangle
            %will be transformed into polygons to adjust with the geometric
            %orientation of the face
            face_polygon = reshape(rectangle' , 1, []);
            
            video_frame = insertShape(video_frame , 'Polygon' , face_polygon, 'LineWidth', 3);
            video_frame = insertMarker(video_frame , xy_points, '+' ,'Color', 'White');
        end
    else 
        % From here,Track the points from frame to frame, and use estimateGeometricTransform2D function to estimate the motion 
        %of the face.Make a copy of the points to be used for computing the geometric transformation between 
        %the points in the previous and the current frames
        
        %Track the points. Note that some points may be lost. so we search their presence
        [xy_points, isFound] = step(point_tracker, gray_frame);
        
        new_points = xy_points(isFound , :); %stores first row of xy_points 
        old_points = previous_points(isFound, :);
        
        nb_of_points = size(new_points, 1);
        
        if nb_of_points >=10
            %estimer la transformation géométrique
            [xform, old_points, new_points] = estimateGeometricTransform(...
                old_points, new_points, 'similarity' ,'MaxDistance' , 4);
            
            %Apply the transformation to the bounding box points
            rectangle = transformPointsForward(xform , rectangle);
            %xform is where the information is installed, ractangle is
            %where to perform the transformation
            
            face_polygon = reshape(rectangle', 1, []);
            
            video_frame = insertShape(video_frame, 'Polygon',face_polygon, 'LineWidth', 3);
            video_frame = insertMarker(video_frame, new_points ,'+', 'Color', 'White');
            
            previous_points = new_points;
            setPoints(point_tracker, previous_points );
        end
    end
    step(video_player , video_frame);
    runing_loop = isOpen(video_player);
            
end

clear cam;
release(video_player);
release(point_tracker);
release(face_detector);