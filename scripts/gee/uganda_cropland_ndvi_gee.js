/**************************************************************************
 * Uganda crop and pasture condition — MODIS NDVI/EVI over cropland and
 * grassland, monthly, February 2000 to the latest available month.
 *
 * Paste into the Earth Engine Code Editor (code.earthengine.google.com)
 * and press Run. Check the printed numbers, then start the export from the
 * Tasks tab. The export runs on Google's servers and can take 30–90 minutes.
 *
 * SENSOR picks the satellite: 'terra' (MOD13Q1, morning pass, from Feb 2000)
 * or 'aqua' (MYD13Q1, afternoon pass, from Jul 2002). Running both lets you
 * check whether a change in the record is real or an artefact of one sensor
 * (Terra's orbit has drifted since 2020).
 *
 * Output: uganda_vegetation_by_region_monthly.csv (Terra) or
 *   uganda_vegetation_by_region_monthly_aqua.csv (Aqua), one row per region,
 *   land-cover class and month:
 *     region, group, cover, date, year, month, ndvi, evi, good_frac, cover_km2
 *   ndvi, evi   average over 250 m pixels of that cover class (good-quality
 *               observations only)
 *   good_frac   share of the class (by area) with a good-quality observation
 *               that month; low values mean cloud, so treat with care
 *   cover_km2   area of the class in the region
 *
 * Regions match the rainfall project (uganda_regional_rainfall_gee.js):
 * Uganda (national), Karamoja, Lake Victoria basin, and the four admin regions.
 * Feeds notebooks/03 in the uganda-food-prices project.
 **************************************************************************/

// ---------------------------------------------------------------- settings
var SENSOR = 'terra';           // 'terra' or 'aqua'
var SENSORS = {
  terra: {collection: 'MODIS/061/MOD13Q1', start: '2000-02-01', suffix: ''},
  aqua:  {collection: 'MODIS/061/MYD13Q1', start: '2002-07-01', suffix: '_aqua'}
};
var START = SENSORS[SENSOR].start;
// Stop a month short of today, so the last month isn't half-filled while MODIS
// composites are still being processed (they appear 2–4 weeks after collection)
var END   = ee.Date(Date.now()).advance(-1, 'month');
// If the export fails with a memory or time error, raise TILE_SCALE to 8 or 16.
var TILE_SCALE = 4;

// Land-cover classes from ESA WorldCover 2021 (10 m):
// 40 = cropland, 30 = grassland, 20 = shrubland. Grassland and shrubland
// together stand in for pasture, which matters for Karamoja's livestock.
var COVERS = {
  cropland: [40],
  pasture: [20, 30]
};

// ---------------------------------------------------------------- boundaries
var gaul0 = ee.FeatureCollection('FAO/GAUL/2015/level0');
var gaul1 = ee.FeatureCollection('FAO/GAUL/2015/level1');
var uganda = gaul0.filter(ee.Filter.eq('ADM0_NAME', 'Uganda'));
var ugGeom = uganda.geometry();

// Karamoja: FAO GAUL 2015 level-1 districts, as in the rainfall script
var KARAMOJA_DISTRICTS = ['Abim', 'Amudat', 'Kaabong', 'Karenga', 'Kotido',
                          'Moroto', 'Nabilatuk', 'Nakapiripirit', 'Napak'];
var karamojaDistricts = gaul1.filter(ee.Filter.eq('ADM0_NAME', 'Uganda'))
  .filter(ee.Filter.inList('ADM1_NAME', KARAMOJA_DISTRICTS));
var karamoja = ee.Feature(karamojaDistricts.geometry().dissolve(1000))
  .set({region: 'Karamoja', group: 'Focus region'});

// Lake Victoria basin, Uganda part: HydroBASINS level 6 upstream of the Jinja outlet
var basins = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_6')
  .filterBounds(ee.Geometry.Rectangle([28, -5, 37, 3]));
var seedIds = basins.filterBounds(ee.Geometry.Point([33.19, 0.42])).aggregate_array('HYBAS_ID');
var lvIds = ee.List(ee.List.sequence(1, 60).iterate(function (_, acc) {
  acc = ee.List(acc);
  var up = basins.filter(ee.Filter.inList('NEXT_DOWN', acc)).aggregate_array('HYBAS_ID');
  return acc.cat(up).distinct();
}, seedIds)).removeAll(seedIds);
var lakeVictoria = ee.Feature(basins.filter(ee.Filter.inList('HYBAS_ID', lvIds))
  .geometry().dissolve(1000).intersection(ugGeom, 1000))
  .set({region: 'Lake Victoria basin', group: 'Focus region'});

// Admin regions: geoBoundaries level 1 (Central, Eastern, Northern, Western).
// Names arrive as e.g. "Northern Region"; keep just "Northern".
var admin = ee.FeatureCollection('WM/geoLab/geoBoundaries/600/ADM1')
  .filter(ee.Filter.eq('shapeGroup', 'UGA'))
  .map(function (f) {
    return ee.Feature(f.geometry()).set({
      region: ee.String(f.get('shapeName')).replace(' Region', ''),
      group: 'Admin region'
    });
  });

var national = ee.Feature(ugGeom).set({region: 'Uganda (national)', group: 'Reference'});
var regions = ee.FeatureCollection([national, karamoja, lakeVictoria]).merge(admin);
print('Regions (expect 7):', regions.aggregate_array('region'));

// ---------------------------------------------------------------- vegetation
var modis = ee.ImageCollection(SENSORS[SENSOR].collection).filterDate(START, END);
print('Sensor:', SENSOR, SENSORS[SENSOR].collection);
var proj = modis.first().select('NDVI').projection();

var SCALE = proj.nominalScale();   // about 232 m

// Land-cover mask from 10 m WorldCover. Each 250 m pixel takes the class Earth
// Engine assigns at that scale (from its pre-built lower-resolution versions of
// the map), so no expensive 10 m processing runs inside the monthly loop. That
// keeps the export within free-tier compute limits. Averages over thousands of
// pixels per region are not sensitive to this choice.
var worldcover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
function coverMask(classes) {
  return worldcover.remap(classes, ee.List.repeat(1, classes.length), 0).rename('w');
}
var fractions = {
  cropland: coverMask(COVERS.cropland),
  pasture: coverMask(COVERS.pasture)
};

// Quality: SummaryQA 0 = good, 1 = marginal (kept); 2 = snow/ice, 3 = cloudy (dropped)
function prep(img) {
  var good = img.select('SummaryQA').lte(1);
  var vi = img.select(['NDVI', 'EVI']).multiply(0.0001).updateMask(good);
  return ee.Image(vi.addBands(good.rename('good'))
    .copyProperties(img, ['system:time_start']));
}
var vi = modis.map(prep);

// Monthly composites: average of the 16-day composites starting in each month
var start = ee.Date(START);
var nMonths = END.difference(start, 'month').floor();
var months = ee.List.sequence(0, nMonths.subtract(1));

// Cover area per region, attached once (km²)
function withArea(cover) {
  var area = fractions[cover].multiply(ee.Image.pixelArea()).divide(1e6).rename('cover_km2');
  return area.reduceRegions({collection: regions, reducer: ee.Reducer.sum().setOutputs(['cover_km2']),
                             scale: SCALE, crs: proj, tileScale: TILE_SCALE});
}
var regionsByCover = {cropland: withArea('cropland'), pasture: withArea('pasture')};
print('Cropland area by region (km²):', regionsByCover.cropland.aggregate_array('region'),
      regionsByCover.cropland.aggregate_array('cover_km2'));

function monthlyTable(cover, monthIndexes) {
  var w = fractions[cover];
  return ee.FeatureCollection(ee.List(monthIndexes).map(function (i) {
    var m0 = start.advance(ee.Number(i), 'month');
    var m1 = m0.advance(1, 'month');
    var comps = vi.filterDate(m0, m1);
    // An empty month would produce an image with no bands; give it masked bands instead
    comps = ee.ImageCollection(ee.Algorithms.If(comps.size().gt(0), comps,
      ee.ImageCollection([ee.Image.constant([0, 0, 0]).rename(['NDVI', 'EVI', 'good']).updateMask(0)])));
    var mean = comps.select(['NDVI', 'EVI']).mean();
    // good = 1 where at least one good observation that month (0 otherwise)
    var good = comps.select('good').max().unmask(0);
    // Keep only pixels of this cover class
    var img = mean.updateMask(w)
      .addBands(good.rename('good_frac').updateMask(w));
    var stats = img.reduceRegions({collection: regionsByCover[cover], reducer: ee.Reducer.mean(),
                                   scale: SCALE, crs: proj, tileScale: TILE_SCALE});
    return stats.map(function (f) {
      return ee.Feature(null, {
        region: f.get('region'), group: f.get('group'), cover: cover,
        date: m0.format('YYYY-MM'), year: m0.get('year'), month: m0.get('month'),
        ndvi: f.get('NDVI'), evi: f.get('EVI'), good_frac: f.get('good_frac'),
        cover_km2: f.get('cover_km2')
      });
    }).set('n', comps.size());
  })).flatten();
}

var table = monthlyTable('cropland', months).merge(monthlyTable('pasture', months));

// ---------------------------------------------------------------- checks
print('Months to export:', nMonths);
// Preview one recent month only, so the Code Editor doesn't compute all of them
print('Preview, cropland, one recent month:', monthlyTable('cropland', [nMonths.subtract(13)]));
Map.centerObject(uganda, 7);
Map.addLayer(fractions.cropland.selfMask(), {palette: ['16825a']}, 'Cropland (WorldCover 2021)');
Map.addLayer(regions.style({color: '333333', fillColor: '00000000', width: 1}), {}, 'Regions');

// ---------------------------------------------------------------- export
Export.table.toDrive({
  collection: table,
  description: 'uganda_vegetation_by_region_monthly' + SENSORS[SENSOR].suffix,
  fileNamePrefix: 'uganda_vegetation_by_region_monthly' + SENSORS[SENSOR].suffix,
  fileFormat: 'CSV',
  selectors: ['region', 'group', 'cover', 'date', 'year', 'month', 'ndvi', 'evi', 'good_frac', 'cover_km2']
});
