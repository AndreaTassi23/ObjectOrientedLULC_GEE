 /*
This code was developed within the research Tassi A., Vizzari M., 2020, 
“Object-oriented LULC classification in Google Earth Engine combining SNIC, GLCM, and Machine Learning algorithms”, 
Remote Sens. 2020, 12(22), 3776; https://doi.org/10.3390/rs12223776. Please, always refer to this paper for any use. Any suggestion useful for code improvement is very welcome!
Please, send any suggestion or information request to Andrea Tassi (andreatassi23@gmail.com) or Marco Vizzari (marco.vizzari@unipg.it)
*/

/*
CLASSIFICATION
1)Input requirements

  - roi: region of interest
  
  - newfc: feature collection containing all the training data

  - valpnts: validation points randomly generated and manually labelled used to assess the output accuracy

  - dataset: previously generated dataset using the following code: "https://code.earthengine.google.com/3b02d59f8dd400c450e380cc830247a2"

  
2) Pixel-based Approach
    
    Perfoms a pixel-based classification using the bands selected between those available in the dataset previosly generated 
    bands= dataset.bandNames()
    
  
3) Object-oriented Approach
  
   Perfoms an object-oriented approach using the same band selected in the previous approach
  
  - The user can be set the variable "size_segmentation" that is the superpixel seed location spacing, in pixels.
  
  - Define GLCM indices from which we will select the desired additional feature that are the input of PCA 
  
  - Prediction bands used for the classification. 
    This are the mean bands of the input based on label "clusters" and are based on the feature PC1 select from PCA based on the GLCM features input
    
  
*/
 
//----Printing parameters:
//Parameters to allow a proper rendering of the output 

//RGB images 
Map.centerObject(roi,12);
Map.addLayer(dataset, {min: 0.0,max: 0.3,bands: ['B4', 'B3', 'B2'], }, 'RGB');

//Palette for the classification
var palette = [ 
  '3399FF', //(0)  Water    
  '999999', //(1)  Built-up           
  '990000', //(2)  Permanente crop     
  'FAF3C0', ///3)  Annual crops 
  'FA9C44', //(4)  Riparian vegetation and shrubs           
  '80FF00', //(5)  Grassland 
  '006600', //(6)  Woodlands
];  
// name of the legend
var names = ['Water','Built-up','Permanent crops','Annual crops','Riparian vegetation and shrubs','Grassland','Woodlands'];

// set position of panel
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px'
  }
});
 
// Create legend title
var legendTitle = ui.Label({
  value: 'Legend',
  style: {
    fontWeight: 'bold',
    fontSize: '18px',
    margin: '0 0 4px 0',
    padding: '0'
    }
});

// Add the title to the panel
legend.add(legendTitle);
 
// Creates and styles 1 row of the legend.
var makeRow = function(color, name) {
 
      // Create the label that is actually the colored box.
      var colorBox = ui.Label({
        style: {
          backgroundColor: '#' + color,
          // Use padding to give the box height and width.
          padding: '8px',
          margin: '0 0 4px 0'
        }
      });
 
      // Create the label filled with the description text.
      var description = ui.Label({
        value: name,
        style: {margin: '0 0 4px 6px'}
      });
 
      // return the panel
      return ui.Panel({
        widgets: [colorBox, description],
        layout: ui.Panel.Layout.Flow('horizontal')
      });
};
// Add color and and names
for (var i = 0; i < 7; i++) {
  legend.add(makeRow(palette[i], names[i]));
  }  
// add legend to map (alternatively you can also print the legend to the console)
Map.add(legend);

//1) ----Training data
//Creation of the "newfc" feature collection using the pixels having a feature property called "LULC" 
//To improve the information using a buffer with a fixed radius  ( radius = 10 m)
var buffer = function(feature) {
return feature.buffer(10)};
newfc = newfc.map(buffer)

//2) ---- Define the classifier
//if you want use RandomForest (classifier_alg= "RF") else use SVM (classifier_alg= "SVM")
var classifier_alg="RF" 

//If you decided to use the SVM algorithm it's mandatory the normalization of the input bands
if(classifier_alg =="SVM"){
var image = ee.Image(dataset);
// calculate the min and max value of an image
var minMax = image.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: roi,
  scale: 10,
  maxPixels: 10e9,
}); 
// use unit scale to normalize the pixel values
var dataset = ee.ImageCollection.fromImages(
  image.bandNames().map(function(name){
    name = ee.String(name);
    var band = image.select(name);
    return band.unitScale(ee.Number(minMax.get(name.cat('_min'))), ee.Number(minMax.get(name.cat('_max'))))
})).toBands().rename(image.bandNames());
}

//3) ----- Define the superpixel seed location spacing, in pixels: (5 - 10 - 15 - 20)
var size_segmentation = 15

//----- Define the GLCM indices used in input for the PCA
var glcm_bands= ["gray_asm","gray_contrast","gray_corr","gray_ent","gray_var","gray_idm","gray_savg"]

//-----Pixel-based Approach

//Selecting desired bands
var bands= dataset.bandNames()
// Get the predictors into the table and create a training dataset based on "LULC" property
var training = dataset.select(bands).sampleRegions({
  collection: newfc,
  properties: ['LULC'],
  scale: 10
});
//Training a Random Forest Classifier
if(classifier_alg=="RF"){
  var classifier =  ee.Classifier.smileRandomForest(50).train({
    features: training,
    classProperty: 'LULC',
    inputProperties: bands
  }); 
  
}
else if (classifier_alg=="SVM") {
var classifi = ee.Classifier.libsvm({
  kernelType: 'RBF',
  gamma: 1,
  cost: 10
});
var classifier = classifi.train(training, 'LULC', bands);

}
else{
  print("You need to set your classifier for the Pixel Based approach")
}

//Clip and filter the result of pixel' classification 
var classified = dataset.select(bands).classify(classifier).clip(roi).focal_mode();
//Visualize the result
Map.addLayer(classified, {min: 0, max: 6, palette: palette}, 'LULC PIXEL APPROACH', false);


//Create the confusion matrix and calculate the overall accuracy on the training data
//print('RF error matrix_training: ', classifier.confusionMatrix());
//print('RF accuracy_training: ', classifier.confusionMatrix().accuracy());

//Validation of the pixel-based approach
var classifierTest = dataset.select(bands).sampleRegions({
  collection: valpnts,
  properties: ['LULC'],
  scale: 10
});
var classified_test_RF = classifierTest.classify(classifier);
var testAccuracy = classified_test_RF.errorMatrix('LULC', 'classification');
//print('Pixel approach_Test confusion matrix: ', testAccuracy);  
print('PIXEL APPROACH : Overall Accuracy ', testAccuracy.accuracy());

//Print the number of pixels for each class 
var analysis_image = classified.select("classification")

var class_1 =  analysis_image.updateMask(analysis_image.eq(0))
var class_2 =  analysis_image.updateMask(analysis_image.eq(1))
var class_3 =  analysis_image.updateMask(analysis_image.eq(2))
var class_4 =  analysis_image.updateMask(analysis_image.eq(3))
var class_5 =  analysis_image.updateMask(analysis_image.eq(4))
var class_6 =  analysis_image.updateMask(analysis_image.eq(5))
var class_7 =  analysis_image.updateMask(analysis_image.eq(6))

var all = class_1.addBands(class_2).addBands(class_3).addBands(class_4).addBands(class_5).addBands(class_6).addBands(class_7)

var count_pixels_one = all.reduceRegion({
  reducer: ee.Reducer.count(),
  geometry: roi,
  scale:10,
  maxPixels: 1e11,

  })
print(count_pixels_one, "PIXEL APPROACH: pixels for each class")

//-----Object-based Approach

// Segmentation using a SNIC approach based on the dataset previosly generated
var seeds = ee.Algorithms.Image.Segmentation.seedGrid(size_segmentation);

var snic = ee.Algorithms.Image.Segmentation.SNIC({
  image: dataset, 
  compactness: 0,  
  connectivity: 8, 
  neighborhoodSize: 256, 
  seeds: seeds
})

//Create and rescale a grayscale image for GLCM 
var gray = dataset.expression(
      '(0.3 * NIR) + (0.59 * R) + (0.11 * G)', {
      'NIR': dataset.select('B8'),
      'R': dataset.select('B4'),
      'G': dataset.select('B3')
}).rename('gray');

// the glcmTexture size (in pixel) can be adjusted considering the spatial resolution and the object textural characteristics

var glcm = gray.unitScale(0, 0.30).multiply(100).toInt().glcmTexture({size: 2});

//--- Before the PCA the glcm bands are scaled
var image = glcm.select(glcm_bands);
// calculate the min and max value of an image
var minMax = image.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: roi,
  scale: 3,
  maxPixels: 10e9,
}); 
var glcm = ee.ImageCollection.fromImages(
  image.bandNames().map(function(name){
    name = ee.String(name);
    var band = image.select(name);
    return band.unitScale(ee.Number(minMax.get(name.cat('_min'))), ee.Number(minMax.get(name.cat('_max'))))
})).toBands().rename(image.bandNames());


//---- Apply the PCA
// The code relating to the PCA was adapted from the GEE documentation  https://developers.google.com/earth-engine/guides/arrays_eigen_analysis

// Get some information about the input to be used later.
var scale = glcm.projection().nominalScale();
var bandNames = glcm.bandNames();

// Mean center the data to enable a faster covariance reducer and an SD stretch of the principal components.
var meanDict = glcm.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi, 
    scale: scale,
    maxPixels: 10e16
});
var means = ee.Image.constant(meanDict.values(bandNames));
var centered = glcm.subtract(means);

// This helper function returns a list of new band names.
var getNewBandNames = function(prefix) {
  var seq = ee.List.sequence(1, bandNames.length());
  return seq.map(function(b) {
    return ee.String(prefix).cat(ee.Number(b).int());
  });
};

// This function accepts mean centered imagery, a scale and a region in which to perform the analysis. 
// It returns the Principal Components (PC) in the region as a new image.
var getPrincipalComponents = function(centered, scale, region) {
  // Collapse the bands of the image into a 1D array per pixel.
  var arrays = centered.toArray();
  
  // Compute the covariance of the bands within the region.
  var covar = arrays.reduceRegion({
    reducer: ee.Reducer.centeredCovariance(),
    geometry: region,
    scale: scale, 
  });
  
  // Get the 'array' covariance result and cast to an array.
  // This represents the band-to-band covariance within the region.
  var covarArray = ee.Array(covar.get('array'));
  
  // Perform an eigen analysis and slice apart the values and vectors.
  var eigens = covarArray.eigen();
  
  // This is a P-length vector of Eigenvalues.
  var eigenValues = eigens.slice(1, 0, 1);
  // This is a PxP matrix with eigenvectors in rows.
  var eigenVectors = eigens.slice(1, 1);
    
  // Convert the array image to 2D arrays for matrix computations.
  var arrayImage = arrays.toArray(1);
    
  // Left multiply the image array by the matrix of eigenvectors.
  var principalComponents = ee.Image(eigenVectors).matrixMultiply(arrayImage);
    
  // Turn the square roots of the Eigenvalues into a P-band image.
  var sdImage = ee.Image(eigenValues.sqrt())
    .arrayProject([0]).arrayFlatten([getNewBandNames('sd')]);
  
  // Turn the PCs into a P-band image, normalized by SD.
  return principalComponents
    // Throw out an an unneeded dimension, [[]] -> [].
    .arrayProject([0])
    // Make the one band array image a multi-band image, [] -> image.
    .arrayFlatten([getNewBandNames('pc')])
    // Normalize the PCs by their SDs.
    .divide(sdImage);
};

// Get the PCs at the specified scale and in the specified region
var pcImage = getPrincipalComponents(centered, scale, roi);

//Select the band "clusters" from the snic output fixed on its scale of 10 meters and add them the PC1 taken from the PCA data.
// Calculate the mean for each segment with respect to the pixels in that cluster
var clusters_snic = snic.select("clusters")
clusters_snic = clusters_snic.reproject ({crs: clusters_snic.projection (), scale: 10});
//Map.addLayer(clusters_snic.randomVisualizer(), {}, 'clusters')

var new_feature = clusters_snic.addBands(pcImage.select("pc1"))

var new_feature_mean = new_feature.reduceConnectedComponents({
  reducer: ee.Reducer.mean(),
  labelBand: 'clusters'
})

//Create a dataset with the new band used so far together with the band "clusters" and their new mean parameters
var final_bands = new_feature_mean.addBands(snic) 

//Define the training bands removing just the "clusters" bands
var predictionBands=final_bands.bandNames().remove("clusters")

//---- Normalize all data if you decide to use the SVM classifier
if (classifier_alg=="SVM"){
  
var minMax = final_bands.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: roi,
  scale: 10,
  maxPixels: 10e9,
});
var final_bands = ee.ImageCollection.fromImages(
  final_bands.bandNames().map(function(name){
    name = ee.String(name);
    var band = final_bands.select(name);
    return band.unitScale(ee.Number(minMax.get(name.cat('_min'))), ee.Number(minMax.get(name.cat('_max'))))
})).toBands().rename(final_bands.bandNames());
}

//Classification using the classifier with the training bands called predictionBands
var training_geobia = final_bands.select(predictionBands).sampleRegions({
  collection: newfc,
  properties: ['LULC'],
  scale: 10
});

//Training the classifier
if(classifier_alg=="RF"){
  var RF = ee.Classifier.smileRandomForest(50).train({
  features:training_geobia, 
  classProperty:'LULC', 
  inputProperties: predictionBands
});
}
else if (classifier_alg=="SVM") {
var classifie = ee.Classifier.libsvm({
  kernelType: 'RBF',
  gamma: 1,
  cost: 10
});
var RF = classifie.train(training_geobia, 'LULC', predictionBands);
}
else{
  print("You need to set your classifier for the Object based approach")
}

var classy_RF = final_bands.select(predictionBands).classify(RF);
classy_RF = classy_RF.reproject ({crs: classy_RF.projection (), scale: 10});
Map.addLayer(classy_RF.clip(roi), {min: 0, max: 6, palette: palette}, 'LULC GEOBIA APPROACH', true);

//Validation of the object-oriented approach
var classifier_geobia = final_bands.select(predictionBands).sampleRegions({
  collection: valpnts,
  properties: ['LULC'],
  scale: 10
});
var classificazione = classifier_geobia.classify(RF);
var testAccuracy = classificazione.errorMatrix('LULC', 'classification');
//print('GEOBIA approach_Test confusion matrix: ', testAccuracy);  
print('GEOBIA APPROACH: Overall accuracy ', testAccuracy.accuracy());

//Print the number of pixels for each class 
var analysis_image_sl = classy_RF.select("classification")

var class1 =  analysis_image_sl.updateMask(analysis_image_sl.eq(0))
var class2 =  analysis_image_sl.updateMask(analysis_image_sl.eq(1))
var class3 =  analysis_image_sl.updateMask(analysis_image_sl.eq(2))
var class4 =  analysis_image_sl.updateMask(analysis_image_sl.eq(3))
var class5 =  analysis_image_sl.updateMask(analysis_image_sl.eq(4))
var class6 =  analysis_image_sl.updateMask(analysis_image_sl.eq(5))
var class7 =  analysis_image_sl.updateMask(analysis_image_sl.eq(6))

var all = class1.addBands(class2).addBands(class3).addBands(class4).addBands(class5).addBands(class6).addBands(class7)

var count_pixels = all.reduceRegion({
  reducer: ee.Reducer.count(),
  geometry: roi,
  scale:10,
  maxPixels: 1e11,
  })

print(count_pixels, "GEOBIA APPROACH: pixels for each class")
