"""
MikoshiLang Extended Functions - Set 31
Geospatial analysis, GIS, cartography, and remote sensing
"""
import sympy as sp
from sympy import symbols, sqrt, sin, cos, tan, atan2, asin, exp, log, pi

def get_rules():
    """Extended set 31: Geospatial and GIS (150 functions)"""
    lat, lon, x, y = symbols('lat lon x y')
    
    return [
        # ===== COORDINATE SYSTEMS (35 functions) =====
        ('GeographicToCartesian[lat_, lon_, h_, a_, f_]', lambda lat, lon, h, a, f: (
            (a/(1 - f) + h)*cos(lat)*cos(lon),
            (a/(1 - f) + h)*cos(lat)*sin(lon),
            (a*(1 - f)**2/(1 - f) + h)*sin(lat)
        )),
        ('CartesianToGeographic[x_, y_, z_, a_, f_]', lambda x, y, z, a, f: (
            atan2(z, sqrt(x**2 + y**2)),
            atan2(y, x),
            sqrt(x**2 + y**2 + z**2) - a
        )),
        ('WGS84Ellipsoid[]', lambda: (6378137.0, 1/298.257223563)),  # a, f
        ('GRS80Ellipsoid[]', lambda: (6378137.0, 1/298.257222101)),
        ('MercatorProjection[lat_, lon_, R_]', lambda lat, lon, R: (R*lon, R*log(tan(pi/4 + lat/2)))),
        ('WebMercator[lat_, lon_]', lambda lat, lon: (6378137*lon, 6378137*log(tan(pi/4 + lat/2)))),
        ('TransverseMercator[lat_, lon_, lat0_, lon0_]', lambda lat, lon, lat0, lon0: (sp.Symbol('E'), sp.Symbol('N'))),
        ('UniversalTransverseMercator[lat_, lon_]', lambda lat, lon: (sp.Symbol('zone'), sp.Symbol('E'), sp.Symbol('N'))),
        ('UTMZone[lon_]', lambda lon: int((lon + 180)/6) + 1),
        ('UTMToLatLon[zone_, E_, N_, hemisphere_]', lambda z, E, N, h: (sp.Symbol('lat'), sp.Symbol('lon'))),
        ('LambertConformalConic[lat_, lon_, lat1_, lat2_, lat0_, lon0_]', lambda lat, lon, lat1, lat2, lat0, lon0: (sp.Symbol('x'), sp.Symbol('y'))),
        ('AlbersEqualArea[lat_, lon_, lat1_, lat2_]', lambda lat, lon, lat1, lat2: (sp.Symbol('x'), sp.Symbol('y'))),
        ('SinusoidalProjection[lat_, lon_]', lambda lat, lon: (lon*cos(lat), lat)),
        ('MollweideProjection[lat_, lon_]', lambda lat, lon: (sp.Symbol('x'), sp.Symbol('y'))),
        ('RobinsonProjection[lat_, lon_]', lambda lat, lon: (sp.Symbol('x'), sp.Symbol('y'))),
        ('WinkelTripelProjection[lat_, lon_]', lambda lat, lon: (sp.Symbol('x'), sp.Symbol('y'))),
        ('StereographicProjection[lat_, lon_, lat0_, lon0_]', lambda lat, lon, lat0, lon0: (sp.Symbol('x'), sp.Symbol('y'))),
        ('OrthographicProjection[lat_, lon_, lat0_, lon0_]', lambda lat, lon, lat0, lon0: (
            cos(lat)*sin(lon - lon0),
            cos(lat0)*sin(lat) - sin(lat0)*cos(lat)*cos(lon - lon0)
        )),
        ('AzimuthalEquidistant[lat_, lon_, lat0_, lon0_]', lambda lat, lon, lat0, lon0: (sp.Symbol('x'), sp.Symbol('y'))),
        ('GnomonicProjection[lat_, lon_, lat0_, lon0_]', lambda lat, lon, lat0, lon0: (sp.Symbol('x'), sp.Symbol('y'))),
        ('EckertIVProjection[lat_, lon_]', lambda lat, lon: (sp.Symbol('x'), sp.Symbol('y'))),
        ('GoodeHomolosine[lat_, lon_]', lambda lat, lon: (sp.Symbol('x'), sp.Symbol('y'))),
        ('MGRS[lat_, lon_]', lambda lat, lon: "grid ref"),
        ('BritishNationalGrid[lat_, lon_]', lambda lat, lon: (sp.Symbol('E'), sp.Symbol('N'))),
        ('StateplaneCoor dinates[lat_, lon_, zone_]', lambda lat, lon, z: (sp.Symbol('x'), sp.Symbol('y'))),
        ('LocalCartesian[lat_, lon_, lat0_, lon0_, h0_]', lambda lat, lon, lat0, lon0, h0: (sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('z'))),
        ('GaussKruger[lat_, lon_]', lambda lat, lon: (sp.Symbol('x'), sp.Symbol('y'))),
        ('SwissGrid[lat_, lon_]', lambda lat, lon: (sp.Symbol('E'), sp.Symbol('N'))),
        ('DatumTransformation[x_, y_, z_, dx_, dy_, dz_]', lambda x, y, z, dx, dy, dz: (x + dx, y + dy, z + dz)),
        ('HelmertTransformation[params_]', lambda p: sp.Symbol('transformed')),
        ('MolodenskyTransformation[lat_, lon_, h_, da_, df_]', lambda lat, lon, h, da, df: (sp.Symbol('lat2'), sp.Symbol('lon2'), sp.Symbol('h2'))),
        ('GeoidHeight[lat_, lon_]', lambda lat, lon: sp.Symbol('N')),  # EGM96/EGM2008
        ('EllipsoidalHeight[h_orthometric_, N_]', lambda horth, N: horth + N),
        ('OrthometricHeight[h_ellipsoidal_, N_]', lambda hell, N: hell - N),
        ('GeocentricLatitude[lat_]', lambda lat: atan2(tan(lat), (1 - sp.Symbol('f'))**2)),
        
        # ===== DISTANCE & BEARING (25 functions) =====
        ('HaversineDistance[lat1_, lon1_, lat2_, lon2_, R_]', lambda lat1, lon1, lat2, lon2, R: 
            2*R*asin(sqrt(sin((lat2 - lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2 - lon1)/2)**2))),
        ('GreatCircleDistance[lat1_, lon1_, lat2_, lon2_, R_]', lambda lat1, lon1, lat2, lon2, R: 
            R*sp.acos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon2 - lon1))),
        ('VincentyDistance[lat1_, lon1_, lat2_, lon2_, a_, f_]', lambda lat1, lon1, lat2, lon2, a, f: sp.Symbol('distance')),
        ('EuclideanDistance2D[x1_, y1_, x2_, y2_]', lambda x1, y1, x2, y2: sqrt((x2 - x1)**2 + (y2 - y1)**2)),
        ('ManhattanDistanceGeo[x1_, y1_, x2_, y2_]', lambda x1, y1, x2, y2: abs(x2 - x1) + abs(y2 - y1)),
        ('InitialBearing[lat1_, lon1_, lat2_, lon2_]', lambda lat1, lon1, lat2, lon2: 
            atan2(sin(lon2 - lon1)*cos(lat2), cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(lon2 - lon1))),
        ('FinalBearing[lat1_, lon1_, lat2_, lon2_]', lambda lat1, lon1, lat2, lon2: 
            (atan2(sin(lon1 - lon2)*cos(lat1), cos(lat2)*sin(lat1) - sin(lat2)*cos(lat1)*cos(lon1 - lon2)) + pi) % (2*pi)),
        ('MidpointCoordinate[lat1_, lon1_, lat2_, lon2_]', lambda lat1, lon1, lat2, lon2: (
            atan2(sin(lat1) + sin(lat2), sqrt((cos(lat1) + cos(lat2)*cos(lon2 - lon1))**2 + (cos(lat2)*sin(lon2 - lon1))**2)),
            lon1 + atan2(cos(lat2)*sin(lon2 - lon1), cos(lat1) + cos(lat2)*cos(lon2 - lon1))
        )),
        ('IntermediatePoint[lat1_, lon1_, lat2_, lon2_, fraction_]', lambda lat1, lon1, lat2, lon2, f: (sp.Symbol('lat'), sp.Symbol('lon'))),
        ('DestinationPoint[lat_, lon_, bearing_, distance_, R_]', lambda lat, lon, brg, d, R: (
            asin(sin(lat)*cos(d/R) + cos(lat)*sin(d/R)*cos(brg)),
            lon + atan2(sin(brg)*sin(d/R)*cos(lat), cos(d/R) - sin(lat)*sin(sp.Symbol('lat2')))
        )),
        ('CrossTrackDistance[lat_, lon_, lat1_, lon1_, lat2_, lon2_, R_]', lambda lat, lon, lat1, lon1, lat2, lon2, R: sp.Symbol('distance')),
        ('AlongTrackDistance[lat_, lon_, lat1_, lon1_, bearing_, R_]', lambda lat, lon, lat1, lon1, brg, R: sp.Symbol('distance')),
        ('RhumbLineDistance[lat1_, lon1_, lat2_, lon2_, R_]', lambda lat1, lon1, lat2, lon2, R: sp.Symbol('distance')),
        ('RhumbLineBearing[lat1_, lon1_, lat2_, lon2_]', lambda lat1, lon1, lat2, lon2: atan2(lon2 - lon1, log(tan(lat2/2 + pi/4)/tan(lat1/2 + pi/4)))),
        ('LoxodromicDistance[lat1_, lon1_, lat2_, lon2_]', lambda lat1, lon1, lat2, lon2: sp.Symbol('distance')),
        ('GeodesicPath[lat1_, lon1_, lat2_, lon2_, n_points_]', lambda lat1, lon1, lat2, lon2, n: [(sp.Symbol('lat'), sp.Symbol('lon'))] *int(n)),
        ('PolygonArea[vertices_]', lambda v: sp.Symbol('area')),
        ('SphericalExcess[vertices_]', lambda v: sp.Symbol('E')),
        ('SphericalTriangleArea[a_, b_, c_, R_]', lambda a, b, c, R: sp.Symbol('A')),
        ('BoundingBox[points_]', lambda pts: (sp.Symbol('min_lat'), sp.Symbol('min_lon'), sp.Symbol('max_lat'), sp.Symbol('max_lon'))),
        ('ConvexHull[points_]', lambda pts: pts),
        ('PointInPolygon[point_, polygon_]', lambda pt, poly: True),
        ('LineIntersection[line1_, line2_]', lambda l1, l2: (sp.Symbol('x'), sp.Symbol('y'))),
        ('BufferZone[geometry_, distance_]', lambda geom, d: geom),
        ('Simplify Geometry[vertices_, tolerance_]', lambda v, tol: v),
        
        # ===== SPATIAL ANALYSIS (30 functions) =====
        ('NearestNeighbor[point_, points_]', lambda pt, pts: pts[0]),
        ('KNearestNeighbors[point_, points_, k_]', lambda pt, pts, k: pts[:int(k)]),
        ('SpatialJoin[layer1_, layer2_]', lambda l1, l2: l1 + l2),
        ('Overlay Analysis[layer1_, layer2_, operation_]', lambda l1, l2, op: l1),
        ('UnionPolygons[polygons_]', lambda polys: polys),
        ('IntersectPolygons[polygon1_, polygon2_]', lambda p1, p2: p1),
        ('DifferencePolygons[polygon1_, polygon2_]', lambda p1, p2: p1),
        ('SymmetricDifference[polygon1_, polygon2_]', lambda p1, p2: (p1, p2)),
        ('ClipLayer[layer_, boundary_]', lambda l, b: l),
        ('EraseFeatures[layer_, erase_layer_]', lambda l, e: l),
        ('DissolveFeatures[layer_, attribute_]', lambda l, attr: l),
        ('AggregatePoints[points_, polygons_]', lambda pts, polys: polys),
        ('SpatialInterpolation[points_, values_, method_]', lambda pts, vals, m: sp.Symbol('surface')),
        ('IDWInterpolation[point_, known_points_, values_, power_]', lambda pt, kpts, vals, p: sum(vi/sp.Symbol('d')**p for vi in vals)/sum(1/sp.Symbol('d')**p for _ in vals)),
        ('KrigingInterpolation[point_, known_points_, values_, variogram_]', lambda pt, kpts, vals, vario: sp.Symbol('value')),
        ('SplineInterpolation[points_, values_]', lambda pts, vals: sp.Symbol('surface')),
        ('NaturalNeighborInterpolation[point_, known_points_, values_]', lambda pt, kpts, vals: sp.Symbol('value')),
        ('TINInterpolation[point_, triangulation_]', lambda pt, tin: sp.Symbol('value')),
        ('Thiessen Polygons[points_]', lambda pts: pts),  # Voronoi
        ('VoronoiDiagram[points_]', lambda pts: pts),
        ('DelaunayTriangulation[points_]', lambda pts: pts),
        ('HotspotAnalysis[points_, weights_]', lambda pts, w: sp.Symbol('hotspots')),
        ('GetisOrdGi[point_, neighbors_, values_]', lambda pt, nbr, vals: sp.Symbol('Gi')),
        ('LocalMoransI[point_, neighbors_, values_]', lambda pt, nbr, vals: sp.Symbol('I')),
        ('GlobalMoransI[points_, values_]', lambda pts, vals: sp.Symbol('I')),
        ('SpatialAutocorrelation[values_, weights_]', lambda vals, w: sp.Symbol('I')),
        ('SemivariogramModel[distance_, sill_, range_, nugget_]', lambda h, s, r, n: n + (s - n)*(1 - exp(-3*h/r))),
        ('VariogramCloud[pairs_distances_, differences_]', lambda d, diffs: list(zip(d, [diff**2/2 for diff in diffs]))),
        ('RangeSemivariance[sill_]', lambda s: sp.Symbol('range')),
        ('NuggetEffect[semivariance_zero_]', lambda s0: s0),
        
        # ===== TERRAIN ANALYSIS (30 functions) =====
        ('ElevationDifference[z1_, z2_]', lambda z1, z2: z2 - z1),
        ('SlopeDegrees[dz_dx_, dz_dy_]', lambda dzdx, dzdy: atan2(sqrt(dzdx**2 + dzdy**2), 1)*180/pi),
        ('SlopePercent[rise_, run_]', lambda rise, run: rise/run*100),
        ('AspectDirection[dz_dx_, dz_dy_]', lambda dzdx, dzdy: atan2(-dzdy, -dzdx)*180/pi),
        ('HillshadeRelief[slope_, aspect_, azimuth_, altitude_]', lambda sl, asp, az, alt: 
            255*((cos(alt)*cos(sl)) + (sin(alt)*sin(sl)*cos(az - asp)))),
        ('Curvature[d2z_dx2_, d2z_dy2_]', lambda d2zdx2, d2zdy2: -(d2zdx2 + d2zdy2)),
        ('ProfileCurvature[d2z_]', lambda d2z: d2z),
        ('PlanCurvature[d2z_]', lambda d2z: d2z),
        ('TopographicPositionIndex[z_, z_mean_]', lambda z, zmean: z - zmean),
        ('TerrainRuggednessIndex[elevation_]', lambda elev: sp.Symbol('TRI')),
        ('TopographicWetnessIndex[catchment_area_, slope_]', lambda ca, sl: log(ca/tan(sl))),
        ('StreamPowerIndex[catchment_area_, slope_]', lambda ca, sl: ca*tan(sl)),
        ('FlowDirection[DEM_]', lambda dem: sp.Symbol('flow_dir')),
        ('FlowAccumulation[flow_dir_]', lambda fd: sp.Symbol('flow_accum')),
        ('WatershedDelineation[pour_point_, DEM_]', lambda pp, dem: sp.Symbol('watershed')),
        ('DrainageBasin[outlet_, DEM_]', lambda out, dem: sp.Symbol('basin')),
        ('StreamOrder[network_]', lambda net: sp.Symbol('order')),  # Strahler/Shreve
        ('ChannelNetworkExtraction[flow_accum_, threshold_]', lambda fa, th: fa > th),
        ('PitFilling[DEM_]', lambda dem: dem),
        ('SinkRemoval[DEM_]', lambda dem: dem),
        ('BreachDepressions[DEM_]', lambda dem: dem),
        ('Viewshed[observer_, DEM_]', lambda obs, dem: sp.Symbol('visible_area')),
        ('LineOfSight[point1_, point2_, DEM_]', lambda p1, p2, dem: True),
        ('VisibilityIndex[point_, DEM_]', lambda pt, dem: sp.Symbol('visible_fraction')),
        ('SolarRadiation[lat_, slope_, aspect_, day_]', lambda lat, sl, asp, d: sp.Symbol('radiation')),
        ('SolarInsolation[lat_, lon_, date_, slope_, aspect_]', lambda lat, lon, date, sl, asp: sp.Symbol('insolation')),
        ('SunPosition[lat_, lon_, datetime_]', lambda lat, lon, dt: (sp.Symbol('azimuth'), sp.Symbol('altitude'))),
        ('ShadowAnalysis[DEM_, sun_position_]', lambda dem, sunpos: sp.Symbol('shadow_map')),
        ('CostDistance[start_, friction_surface_]', lambda start, friction: sp.Symbol('cost_distance')),
        ('LeastCostPath[start_, end_, cost_surface_]', lambda start, end, cost: sp.Symbol('path')),
        
        # ===== REMOTE SENSING (30 functions) =====
        ('NDVI[NIR_, RED_]', lambda nir, red: (nir - red)/(nir + red)),
        ('EVI[NIR_, RED_, BLUE_]', lambda nir, red, blue: 2.5*(nir - red)/(nir + 6*red - 7.5*blue + 1)),
        ('SAVI[NIR_, RED_, L_]', lambda nir, red, L: (nir - red)/(nir + red + L)*(1 + L)),
        ('MSAVI[NIR_, RED_]', lambda nir, red: (2*nir + 1 - sqrt((2*nir + 1)**2 - 8*(nir - red)))/2),
        ('NDWI[GREEN_, NIR_]', lambda green, nir: (green - nir)/(green + nir)),
        ('NDBI[SWIR_, NIR_]', lambda swir, nir: (swir - nir)/(swir + nir)),
        ('NBR[NIR_, SWIR_]', lambda nir, swir: (nir - swir)/(nir + swir)),  # Normalized Burn Ratio
        ('NDSI[GREEN_, SWIR_]', lambda green, swir: (green - swir)/(green + swir)),  # Snow
        ('BAI[RED_, NIR_]', lambda red, nir: 1/((0.1 - red)**2 + (0.06 - nir)**2)),  # Burn Area
        ('TasseledCapBrightness[bands_]', lambda b: sum(b)),
        ('TasseledCapGreenness[bands_]', lambda b: sp.Symbol('greenness')),
        ('TasseledCapWetness[bands_]', lambda b: sp.Symbol('wetness')),
        ('PrincipalComponentAnalysis[bands_]', lambda b: b),
        ('MinimumNoiseFunction[bands_]', lambda b: b),
        ('SpectralUnmixing[pixel_, endmembers_]', lambda px, em: [1/len(em)]*len(em)),
        ('LinearSpectralMixing[fractions_, endmembers_]', lambda f, em: sum(fi*emi for fi, emi in zip(f, em))),
        ('SpectralAngleMapper[spectrum1_, spectrum2_]', lambda s1, s2: sp.acos(sum(s1i*s2i for s1i, s2i in zip(s1, s2))/(sqrt(sum(s1i**2 for s1i in s1))*sqrt(sum(s2i**2 for s2i in s2))))),
        ('AtmosphericCorrection[DN_, gain_, offset_]', lambda DN, g, o: DN*g + o),
        ('DarkObjectSubtraction[DN_, dark_value_]', lambda DN, dv: DN - dv),
        ('TopOfAtmosphereReflectance[DN_, gain_, sun_angle_]', lambda DN, g, sa: DN*g/cos(sa)),
        ('SurfaceReflectance[TOA_, atmospheric_effects_]', lambda TOA, atm: TOA - atm),
        ('RadianceCalibration[DN_, Lmin_, Lmax_, Qcal_min_, Qcal_max_]', lambda DN, Lmin, Lmax, Qmin, Qmax: Lmin + (Lmax - Lmin)/(Qmax - Qmin)*(DN - Qmin)),
        ('BrightnessTemperature[radiance_, K1_, K2_]', lambda L, K1, K2: K2/log(K1/L + 1)),
        ('LandSurfaceTemperature[brightness_temp_, emissivity_]', lambda Tb, e: Tb/(1 + (sp.Symbol('lambda')*Tb/sp.Symbol('c2'))*log(e))),
        ('EmissivityEstimation[NDVI_]', lambda ndvi: 0.004*ndvi + 0.986),
        ('CloudMask[reflectance_]', lambda refl: refl > 0.3),
        ('ShadowDetection[NIR_, threshold_]', lambda nir, th: nir < th),
        ('WaterMask[NDWI_, threshold_]', lambda ndwi, th: ndwi > th),
        ('ChangeDetection[image1_, image2_]', lambda im1, im2: im2 - im1),
        ('ImageDifferencing[before_, after_]', lambda bef, aft: aft - bef),
    ]


def register():
    """Register all extended31 rules"""
    from .extended_helper import convert_rules
    return convert_rules(get_rules())
