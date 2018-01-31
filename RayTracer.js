function setup()
{
	UI = {};
	UI.tabs = [];
	UI.titleLong = 'Ray Tracer';
	UI.titleShort = 'RayTracerSimple';

	UI.tabs.push(
		{
		visible: true,
		type: `x-shader/x-fragment`,
		title: `RaytracingDemoFS - GL`,
		id: `RaytracingDemoFS`,
		initialValue: `

precision highp float;

struct PointLight {
  vec3 position;
  vec3 color;
};

struct Material {
  vec3  diffuse;
  vec3  specular;
  float glossiness;
  float kr;
  float kt;

  // The information required for Reflection and Refraction are the parameters kr and kt respectively. kr is the Reflection Multiplier (Reflection Weight), which determines the amount of light reflected by the material, compared to the original intensity. Its value is typically between 0.0 (for non-reflecting surfaces) and 1.0 (for completely reflecting surfaces), since no natural material exists which gives off more light than what is incident on it. Similarly, kt is the Refraction Multiplier (Refraction Weight) used to determine the amount of refraction that occurs, when the light passes through the material. Due to similar reasons as before, its typical value is between 0.0 (for opaque surface) and 1.0 (for transparent surfaces). The values of these parameters are defined separately for each material, in the getMaterial functions.
};

struct Sphere {
  vec3 position;
  float radius;
  Material material;
};

struct Plane {
  vec3 normal;
  float d;				//The parameter d is the distance of the plane from the origin
  Material material;
};

struct Cylinder {
  vec3 position;
  vec3 direction;  
  float radius;
  Material material;
};

const int lightCount = 2;
const int sphereCount = 3;
const int planeCount = 1;
const int cylinderCount = 1;

struct Scene {
  vec3 ambient;
  PointLight[lightCount] lights;
  Sphere[sphereCount] spheres;
  Plane[planeCount] planes;
  Cylinder[cylinderCount] cylinders;
};

struct Ray {
  vec3 origin;
  vec3 direction;
};

// Contains all information pertaining to a ray/object intersection
struct HitInfo {
  bool hit;
  float t;																		//In the general formula for a ray, P = P0 + vt, t gives the magnitude of the unit vector v.
  vec3 position;
  vec3 normal;
  Material material;
};

HitInfo getEmptyHit() {
  return HitInfo(
    false, 
    0.0, 
    vec3(0.0), 
    vec3(0.0),
    Material(vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0)								//All values are either 0 or false, since the ray has not hit any surface
    );
}

HitInfo intersectSphere(const Ray ray, const Sphere sphere) {
  
    vec3 to_sphere = ray.origin - sphere.position;								//This is the vector between the ray's origin and the centre of the sphere
  
    float a = dot(ray.direction, ray.direction);								//The ray-sphere intersection generates a formula of the form at^2 + bt + c = 0, which is a Quadratic Equation to be 																				 //solved as shown below. By finding the value of t, we can determine the hit position on the sphere, using the 																					//general form of a ray P = P0 + vt.
    float b = 2.0 * dot(ray.direction, to_sphere);
    float c = dot(to_sphere, to_sphere) - sphere.radius * sphere.radius;
    float D = b * b - 4.0 * a * c;
    if (D > 0.0)
    {
		float t0 = (-b - sqrt(D)) / (2.0 * a);
		float t1 = (-b + sqrt(D)) / (2.0 * a);
      	float t = min(t0, t1);
      	vec3 hitPosition = ray.origin + t * ray.direction;
        return HitInfo(
          	true,
          	t,
          	hitPosition,
          	normalize(hitPosition - sphere.position),							//The normal at the hit position is given by the vector between the centre of the sphere, and the hit position.
          	sphere.material);
    }
    return getEmptyHit();
}

HitInfo intersectPlane(const Ray ray,const Plane plane) {						//The ray-plane intersection can be shown by solving for P in the equations
  																				//P = P0 + vt, and 
  																				//P.N + d = 0, and using that to find the value of t. Once t is found, it can be used in the first equation to find 
  																				//the hit position on the plane.
  float ap = dot(ray.origin,plane.normal);
  float bp = dot(ray.direction,plane.normal);
  float tp = -(((ap) + plane.d) / (bp));
  vec3 hitpositionplane = ray.origin + (tp * ray.direction);
  return HitInfo(
  		true,
  		tp,
  		hitpositionplane,
  		normalize(plane.normal),												//The normal is the same unit vector on all points of the lane, which is plane.normal
  		plane.material);
}

float lengthSquared(vec3 x) {
  return dot(x, x);
}

HitInfo intersectCylinder(const Ray ray, const Cylinder cylinder) {			
    vec3 to_cylinder = ray.origin - cylinder.position;							//The ray-cylinder intersection is similar to the ray-sphere intersection, where we have to find the value of the 																					//magnitude t by solving a Quadratic Equation of the form At^2 + Bt + C = 0. The following are the required 																						//calculations for solving Quadratic, taking A = ac, B = bc, and C = cc, with Discriminant D = Dc.
  
    float dotv = dot(ray.direction, cylinder.direction);
  	float dotpv = dot(to_cylinder, cylinder.direction);
  	vec3 roota = (ray.direction - (dotv * cylinder.direction));
  	float ac = dot(roota, roota);
    float bc = 2.0 * (dot(roota,(to_cylinder - (dotpv * cylinder.direction))));
    vec3 rootc = to_cylinder - (dotpv * cylinder.direction);
  	float cc = (dot(rootc, rootc) - (cylinder.radius * cylinder.radius));
    float Dc = bc * bc - 4.0 * ac * cc;
    if (Dc > 0.0)
    {
		float t0c = (-bc - sqrt(Dc)) / (2.0 * ac);
		float t1c = (-bc + sqrt(Dc)) / (2.0 * ac);
      	float tc = min(t0c, t1c);
      	vec3 hitPositionCylinder = ray.origin + tc * ray.direction;
      	vec3 vb = hitPositionCylinder - cylinder.position;						//After finding the hit position, the normal at that point will be the vector between the corresponding point of the 
      																			//axis and the hit position. To find the point on the axis, we can take the projection (projectionb) of the vector 																					//between the cylinder position and the hit position (vb).
      	vec3 projectionb = (dot(cylinder.direction, vb) * cylinder.direction);	
      	vec3 projpos = cylinder.position + projectionb;							//projpos is the point on the axis, corresponding to the hit position.
     	return HitInfo(
          	true,
          	tc,
          	hitPositionCylinder,
          	normalize(hitPositionCylinder - projpos),							//The normal is the vector between the corresponding point of the axis and the hit position
          	cylinder.material);
    }
  	return getEmptyHit();
}
int mat;
HitInfo intersectScene(const Scene scene, const Ray ray, float tMin, float tMax)
{
    HitInfo best_hit_info;
    best_hit_info.t = tMax;
  	best_hit_info.hit = false;

      for (int i = 0; i < cylinderCount; ++i) {
        Cylinder cylinder = scene.cylinders[i];
        HitInfo hit_info = intersectCylinder(ray, cylinder);

        if(	hit_info.hit && 
           	hit_info.t < best_hit_info.t &&
           	hit_info.t > tMin)
        {
            best_hit_info = hit_info;				
        }
    }

    for (int i = 0; i < sphereCount; ++i) {
        Sphere sphere = scene.spheres[i];
        HitInfo hit_info = intersectSphere(ray, sphere);

        if(	hit_info.hit && 
           	hit_info.t < best_hit_info.t &&
           	hit_info.t > tMin)
        {
            best_hit_info = hit_info;
        }
    }

    for (int i = 0; i < planeCount; ++i) {
        Plane plane = scene.planes[i];
        HitInfo hit_info = intersectPlane(ray, plane);

        if(	hit_info.hit && 
           	hit_info.t < best_hit_info.t &&
           	hit_info.t > tMin)
        {
            best_hit_info = hit_info;
        }
    }

  
  return best_hit_info;													//This function returns the structure best_hit_info, which specifies the type of object that the ray hits. This was done by 																		//checking all the types objects one by one, to see which object is hit by the ray first.
}

vec3 shadeFromLight(
  const Scene scene,
  const Ray ray,
  const HitInfo hit_info,
  const PointLight light)
{ 
  vec3 hitToLight = light.position - hit_info.position;
  
  vec3 lightDirection = normalize(hitToLight);
  vec3 viewDirection = normalize(hit_info.position - ray.origin);
  vec3 reflectedDirection = reflect(viewDirection, hit_info.normal);

  float diffuse_term = max(0.0, dot(lightDirection, hit_info.normal));
  float specular_term  = pow(max(0.0, dot(lightDirection, reflectedDirection)), hit_info.material.glossiness);
  float visibility = 5.0;
  // Put your shadow test here
  Ray shadowRay;																//Creating a new ray called shadowRay, which traces the path behind the point where the original light first hits
  shadowRay.origin = hit_info.position;											//shadowRay starts just behind the point where the light ray hits, in the same direction
  shadowRay.direction = lightDirection;
  HitInfo shadeHitInfo = intersectScene(scene, shadowRay, 0.001, 100000.0);		//It is tested to see where it intersects with the scene
  if(shadeHitInfo.hit && shadeHitInfo.t > 0.0)									//If the shadowRay hits any object, the visibility at that point is lowered.
    visibility = 1.0;															//One common mistake is putting the value of visibility to 0. This is not valid, since there is surrounding ambient 																				//light present, which will cause some visibility.
  
  Ray mirrorRay;
  mirrorRay.origin = hit_info.position;
  mirrorRay.direction = reflect(lightDirection, hit_info.normal);
  HitInfo mirrorHitInfo = intersectScene(scene, mirrorRay, 0.001, 100000.0);
     
  return 	visibility * 
    		light.color * (
    		specular_term * hit_info.material.specular +
      		diffuse_term * hit_info.material.diffuse);
}


vec3 background(const Ray ray) {
  // A simple implicit sky that can be used for the background
  return vec3(0.1) + vec3(0.4, 0.5, 0.8) * max(0.0, ray.direction.y);
}

vec3 shade(const Scene scene, const Ray ray, const HitInfo hit_info) {
  
  	if(!hit_info.hit) {
  		return background(ray);
  	}
  
    vec3 shading = scene.ambient * hit_info.material.diffuse;
    for (int i = 0; i < lightCount; ++i) {
        shading += shadeFromLight(scene, ray, hit_info, scene.lights[i]); 
    }
    return shading;
}

Ray getFragCoordRay(const vec2 frag_coord) {
  	float sensorDistance = 1.0;
  	vec2 sensorMin = vec2(-1, -0.5);
  	vec2 sensorMax = vec2(1, 0.5);
  	vec2 pixelSize = (sensorMax- sensorMin) / vec2(800, 400);
  	vec3 origin = vec3(0, 0, sensorDistance);
    vec3 direction = normalize(vec3(sensorMin + pixelSize * frag_coord, -sensorDistance));  
  
  	return Ray(origin, direction);
}

vec3 colorForFragment(const Scene scene, const vec2 fragCoord) {
      
    Ray initialRay = getFragCoordRay(fragCoord);  
  	HitInfo initialHitInfo = intersectScene(scene, initialRay, 0.001, 10000.0);  
  	vec3 result = shade(scene, initialRay, initialHitInfo);
	
  	Ray currentRay;
  	HitInfo currentHitInfo;
  	
  	// Compute the reflection
  	currentRay = initialRay;
  	currentHitInfo = initialHitInfo;
  	float reflectionWeight = currentHitInfo.material.kr;								//The reflection weight if defined by the Reflection Multiplier kr of the sppecific material.
      																					//The computation for reflection is done here. For a maximum of two reflection steps, ray tracing is applied
      																					//in order to trace the path of the ray and to check which surfaces the ray hits.
  	const int maxReflectionStepCount = 2;
  	for(int i = 0; i < maxReflectionStepCount; i++) {
      
      if(!currentHitInfo.hit) break;

      Ray nextRay;																		//nextRay will be the ray which is reflected off from the current surface, from the current ray (currentRay)
      nextRay.direction = reflect(currentRay.direction,currentHitInfo.normal);			//The direction of nextRay can be found using the 'reflect' function, which takes the original ray and the 																							//normal as parameters, to return the reflected ray
      nextRay.origin = currentHitInfo.position;
      
      currentRay = nextRay;																//For the next iteration, the current ray is set to the previous reflected ray (nextRay).
      
      currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);      
            
      result += reflectionWeight * shade(scene, currentRay, currentHitInfo);			//The final result of the fragment is added to the fragment shader, which includes the reflection weight.
    }

  	// Compute the refraction
  	currentRay = initialRay;  
  	currentHitInfo = initialHitInfo;
  	
  	float refractionWeight = currentHitInfo.material.kt;								//Here, refractionWeight is set to the Refraction Multiplier kt of the specific material

  	const int maxRefractionStepCount = 2;												//A max of 2 refraction steps are applied
  	for(int i = 0; i < maxRefractionStepCount; i++) {
      
      if(!currentHitInfo.hit) break;

      Ray nextRay;																		//Similar to reflection, ray tracing is applied by using the predefined function 'refract'.
      nextRay.direction = refract(currentRay.direction,currentHitInfo.normal, 0.9);		//The refracted ray is stored in nextRay
      nextRay.origin = currentHitInfo.position;
	    // Put your code to compute the reflection ray
	  currentRay = nextRay;																//To continue ray tracing, the current ray is set as the refracted ray (nextRay).
      currentHitInfo = intersectScene(scene, currentRay, 0.001, 10000.0);
            
      result += refractionWeight * shade(scene, currentRay, currentHitInfo) / 3.0;
    }

  return result;
}

Material getDefaultMaterial() {															//For details on materials, please refer the included Table
  return Material(vec3(0.3), vec3(0), 1.0, 0.0, 0.0);									//The parameters in each material structure are Diffuse, Specular, Glossiness, Reflection Weight and 																								//Refraction Weight respectively
}

Material getPaperMaterial() {
  return Material(vec3(0.2), vec3(0), 0.5, 0.0, 0.0);
}

Material getPlasticMaterial() {
  return Material(vec3(0.2,0.0,0.02), vec3(0.2,0.1,0.2), 9.0, 0.06, 0.0);
}

Material getGlassMaterial() {
  return Material(vec3(0.1,0.1,0.1), vec3(0.02,0.02,0.02), 0.9, 0.15, 1.0);
}

Material getSteelMirrorMaterial() {
  return Material(vec3(0.02,0.02,0.02), vec3(0.2,0.1,0.2), 9.0, 1.0, 0.0);
}

void main()
{
    // Setup scene
    Scene scene;

  	scene.ambient = vec3(0.12, 0.15, 0.2);
  
    // Lights
    scene.lights[0].position = vec3(5, 15, -5);
    scene.lights[0].color    = 0.5 * vec3(0.8, 0.6, 0.5);

    scene.lights[1].position = vec3(-15, 10, 2);
    scene.lights[1].color    = 0.5 * vec3(0.7, 0.5, 1);

    // Primitives																			//The scene consists of 5 primitives (3 spheres, 1 plane and 1 cylinder). The plane is present at the 																								//bottom, and has a reflective steel mirror material. The cylinder intersects one of the spheres, and 																								//has a paper material. One of the spheres (present on the right) also has a paper material. The second 																							//sphere has plastic material, and looks like a red plastic ball. The last sphere, is a glass ball, with 																							 //light passing through its transparent surface.
    scene.spheres[0].position            	= vec3(6, -2, -12);
    scene.spheres[0].radius              	= 5.0;
    scene.spheres[0].material 				= getPaperMaterial();

    scene.spheres[1].position            	= vec3(-6, -2, -12);
    scene.spheres[1].radius             	= 4.0;
    scene.spheres[1].material				= getPlasticMaterial();

    scene.spheres[2].position            	= vec3(0, 2, -12);
    scene.spheres[2].radius              	= 3.0;
    scene.spheres[2].material   			= getGlassMaterial();

    scene.planes[0].normal            		= vec3(0, 1, 0);
  	scene.planes[0].d              			= 4.5;
    scene.planes[0].material				= getSteelMirrorMaterial();

    scene.cylinders[0].position            	= vec3(0, 2, -10);
  	scene.cylinders[0].direction            = normalize(vec3(1, 3, -1));
  	scene.cylinders[0].radius         		= 0.5;
    scene.cylinders[0].material				= getPaperMaterial();

  // compute color for fragment
  gl_FragColor.rgb = colorForFragment(scene, gl_FragCoord.xy);
  gl_FragColor.a = 1.0;
}
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	UI.tabs.push(
		{
		visible: false,
		type: `x-shader/x-vertex`,
		title: `RaytracingDemoVS - GL`,
		id: `RaytracingDemoVS`,
		initialValue: `attribute vec3 position;

    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;
  
    void main(void) {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`,
		description: ``,
		wrapFunctionStart: ``,
		wrapFunctionEnd: ``
	});

	 return UI; 
}//!setup

var gl;
function initGL(canvas) {
	try {
		gl = canvas.getContext("webgl");
		gl.viewportWidth = canvas.width;
		gl.viewportHeight = canvas.height;
	} catch (e) {
	}
	if (!gl) {
		alert("Could not initialise WebGL, sorry :-(");
	}
}

function getShader(gl, id) {
	var shaderScript = document.getElementById(id);
	if (!shaderScript) {
		return null;
	}

	var str = "";
	var k = shaderScript.firstChild;
	while (k) {
		if (k.nodeType == 3) {
			str += k.textContent;
		}
		k = k.nextSibling;
	}

	var shader;
	if (shaderScript.type == "x-shader/x-fragment") {
		shader = gl.createShader(gl.FRAGMENT_SHADER);
	} else if (shaderScript.type == "x-shader/x-vertex") {
		shader = gl.createShader(gl.VERTEX_SHADER);
	} else {
		return null;
	}

    console.log(str);
	gl.shaderSource(shader, str);
	gl.compileShader(shader);

	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		alert(gl.getShaderInfoLog(shader));
		return null;
	}

	return shader;
}

function RaytracingDemo() {
}

RaytracingDemo.prototype.initShaders = function() {

	this.shaderProgram = gl.createProgram();

	gl.attachShader(this.shaderProgram, getShader(gl, "RaytracingDemoVS"));
	gl.attachShader(this.shaderProgram, getShader(gl, "RaytracingDemoFS"));
	gl.linkProgram(this.shaderProgram);

	if (!gl.getProgramParameter(this.shaderProgram, gl.LINK_STATUS)) {
		alert("Could not initialise shaders");
	}

	gl.useProgram(this.shaderProgram);

	this.shaderProgram.vertexPositionAttribute = gl.getAttribLocation(this.shaderProgram, "position");
	gl.enableVertexAttribArray(this.shaderProgram.vertexPositionAttribute);

	this.shaderProgram.projectionMatrixUniform = gl.getUniformLocation(this.shaderProgram, "projectionMatrix");
	this.shaderProgram.modelviewMatrixUniform = gl.getUniformLocation(this.shaderProgram, "modelViewMatrix");
}

RaytracingDemo.prototype.initBuffers = function() {
	this.triangleVertexPositionBuffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
	
	var vertices = [
		 -1,  -1,  0,
		 -1,  1,  0,
		 1,  1,  0,

		 -1,  -1,  0,
		 1,  -1,  0,
		 1,  1,  0,
	 ];
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
	this.triangleVertexPositionBuffer.itemSize = 3;
	this.triangleVertexPositionBuffer.numItems = 3 * 2;
}

RaytracingDemo.prototype.drawScene = function() {
			
	var perspectiveMatrix = new J3DIMatrix4();	
	perspectiveMatrix.setUniform(gl, this.shaderProgram.projectionMatrixUniform, false);

	var modelViewMatrix = new J3DIMatrix4();	
	modelViewMatrix.setUniform(gl, this.shaderProgram.modelviewMatrixUniform, false);
		
	gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexPositionBuffer);
	gl.vertexAttribPointer(this.shaderProgram.vertexPositionAttribute, this.triangleVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
	
	gl.drawArrays(gl.TRIANGLES, 0, this.triangleVertexPositionBuffer.numItems);
}

RaytracingDemo.prototype.run = function() {
	this.initShaders();
	this.initBuffers();

	gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
	gl.clear(gl.COLOR_BUFFER_BIT);

	this.drawScene();
};

function init() {	
	

	env = new RaytracingDemo();	
	env.run();

    return env;
}

function compute(canvas)
{
    env.initShaders();
    env.initBuffers();

    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clear(gl.COLOR_BUFFER_BIT);

    env.drawScene();
}
