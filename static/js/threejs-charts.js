/**
 * Three.js Chart Components for HAL Frontend
 * Provides interactive 3D visualizations for agent performance data
 */

class ThreeJSCharts {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.animationId = null;
    }

    /**
     * Initialize Three.js scene with camera, renderer, and controls
     */
    init(containerId, width = 800, height = 600) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container with ID ${containerId} not found`);
            return false;
        }

        // Check if Three.js is loaded
        if (typeof THREE === 'undefined') {
            console.error('Three.js is not loaded');
            return false;
        }

        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf8f9fa);

        // Camera setup
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.position.set(50, 50, 50);

        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(width, height);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(this.renderer.domElement);

        // Basic controls (fallback if OrbitControls not available)
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
        } else {
            console.warn('OrbitControls not available, using basic mouse controls');
            this.setupBasicControls();
        }

        // Lighting
        this.setupLighting();

        // Start animation loop
        this.animate();

        return true;
    }

    /**
     * Setup basic mouse controls as fallback
     */
    setupBasicControls() {
        let isRotating = false;
        let lastMouseX = 0;
        let lastMouseY = 0;

        this.renderer.domElement.addEventListener('mousedown', (e) => {
            isRotating = true;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        });

        this.renderer.domElement.addEventListener('mousemove', (e) => {
            if (!isRotating) return;
            
            const deltaX = e.clientX - lastMouseX;
            const deltaY = e.clientY - lastMouseY;
            
            this.camera.position.x = Math.cos(deltaX * 0.01) * 50;
            this.camera.position.z = Math.sin(deltaX * 0.01) * 50;
            this.camera.position.y += deltaY * 0.1;
            
            this.camera.lookAt(0, 0, 0);
            
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        });

        this.renderer.domElement.addEventListener('mouseup', () => {
            isRotating = false;
        });
    }

    /**
     * Setup scene lighting
     */
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);

        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 25);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);

        // Point light
        const pointLight = new THREE.PointLight(0xffffff, 0.4, 100);
        pointLight.position.set(-25, 25, 25);
        this.scene.add(pointLight);
    }

    /**
     * Create 3D scatter plot for agent performance vs cost
     */
    createPerformanceScatter(data, containerId) {
        if (!this.init(containerId)) return;

        const group = new THREE.Group();

        // Color palette for different agents
        const colors = [
            0x3498db, 0xe74c3c, 0x2ecc71, 0xf39c12, 0x9b59b6,
            0x1abc9c, 0xe67e22, 0x34495e, 0x95a5a6, 0xd35400
        ];

        // Create spheres for each data point
        data.forEach((point, index) => {
            const geometry = new THREE.SphereGeometry(1.5, 16, 16);
            const material = new THREE.MeshLambertMaterial({
                color: colors[index % colors.length],
                transparent: true,
                opacity: 0.8
            });

            const sphere = new THREE.Mesh(geometry, material);
            
            // Position based on accuracy, cost, and benchmark (z-axis)
            sphere.position.set(
                (point.accuracy || 0) * 0.5,
                (point.cost || 0) * 10,
                (point.benchmarkIndex || 0) * 10
            );

            sphere.userData = {
                agent: point.agent,
                model: point.model,
                accuracy: point.accuracy,
                cost: point.cost,
                benchmark: point.benchmark
            };

            sphere.castShadow = true;
            sphere.receiveShadow = true;

            group.add(sphere);
        });

        // Add axes
        this.addAxes(group, 50, 25, 25);

        this.scene.add(group);

        // Add interaction
        this.addInteraction();
    }

    /**
     * Create 3D bar chart for benchmark performance
     */
    createBenchmarkBars(highlights, containerId) {
        if (!this.init(containerId)) return;

        const group = new THREE.Group();
        const barSpacing = 8;
        const agentSpacing = 2;

        highlights.forEach((benchmark, benchmarkIndex) => {
            const benchmarkGroup = new THREE.Group();
            
            benchmark.agents.forEach((agent, agentIndex) => {
                // Create bar geometry
                const height = agent.accuracy / 10; // Scale down for better visualization
                const geometry = new THREE.BoxGeometry(1.5, height, 1.5);
                
                // Color based on performance
                const performance = agent.accuracy / 100;
                const color = new THREE.Color().setHSL(performance * 0.3, 0.8, 0.6);
                
                const material = new THREE.MeshLambertMaterial({ color });
                const bar = new THREE.Mesh(geometry, material);

                // Position the bar
                bar.position.set(
                    benchmarkIndex * barSpacing,
                    height / 2,
                    agentIndex * agentSpacing - (benchmark.agents.length * agentSpacing) / 2
                );

                bar.userData = {
                    agent: agent.base_agent,
                    model: agent.model_name,
                    accuracy: agent.accuracy,
                    cost: agent.total_cost,
                    benchmark: benchmark.benchmark
                };

                bar.castShadow = true;
                bar.receiveShadow = true;

                benchmarkGroup.add(bar);
            });

            group.add(benchmarkGroup);
        });

        // Add ground plane
        const planeGeometry = new THREE.PlaneGeometry(200, 100);
        const planeMaterial = new THREE.MeshLambertMaterial({ 
            color: 0xcccccc, 
            transparent: true, 
            opacity: 0.3 
        });
        const plane = new THREE.Mesh(planeGeometry, planeMaterial);
        plane.rotation.x = -Math.PI / 2;
        plane.receiveShadow = true;
        group.add(plane);

        this.scene.add(group);
        this.addInteraction();
    }

    /**
     * Create animated particle system for model comparison
     */
    createModelParticles(modelData, containerId) {
        if (!this.init(containerId)) return;

        const particleCount = modelData.length * 20;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);

        let particleIndex = 0;

        modelData.forEach((model, modelIndex) => {
            const modelColor = new THREE.Color().setHSL(modelIndex / modelData.length, 0.8, 0.6);
            
            for (let i = 0; i < 20; i++) {
                // Create particle cluster around model position
                const radius = 5 + Math.random() * 10;
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.random() * Math.PI;

                positions[particleIndex * 3] = modelIndex * 15 + radius * Math.sin(phi) * Math.cos(theta);
                positions[particleIndex * 3 + 1] = radius * Math.cos(phi);
                positions[particleIndex * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);

                colors[particleIndex * 3] = modelColor.r;
                colors[particleIndex * 3 + 1] = modelColor.g;
                colors[particleIndex * 3 + 2] = modelColor.b;

                particleIndex++;
            }
        });

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 3,
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        });

        const particles = new THREE.Points(geometry, material);
        this.scene.add(particles);

        // Animate particles
        const animate = () => {
            particles.rotation.y += 0.005;
        };

        this.addCustomAnimation(animate);
    }

    /**
     * Add coordinate axes to the scene
     */
    addAxes(group, xLength, yLength, zLength) {
        // X-axis (red)
        const xGeometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(xLength, 0, 0)
        ]);
        const xMaterial = new THREE.LineBasicMaterial({ color: 0xff0000 });
        const xAxis = new THREE.Line(xGeometry, xMaterial);
        group.add(xAxis);

        // Y-axis (green)
        const yGeometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, yLength, 0)
        ]);
        const yMaterial = new THREE.LineBasicMaterial({ color: 0x00ff00 });
        const yAxis = new THREE.Line(yGeometry, yMaterial);
        group.add(yAxis);

        // Z-axis (blue)
        const zGeometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, 0, zLength)
        ]);
        const zMaterial = new THREE.LineBasicMaterial({ color: 0x0000ff });
        const zAxis = new THREE.Line(zGeometry, zMaterial);
        group.add(zAxis);
    }

    /**
     * Add mouse interaction for tooltips
     */
    addInteraction() {
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const tooltip = this.createTooltip();

        const onMouseMove = (event) => {
            const rect = this.renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, this.camera);
            const intersects = raycaster.intersectObjects(this.scene.children, true);

            if (intersects.length > 0 && intersects[0].object.userData) {
                const data = intersects[0].object.userData;
                this.showTooltip(tooltip, event, data);
            } else {
                this.hideTooltip(tooltip);
            }
        };

        this.renderer.domElement.addEventListener('mousemove', onMouseMove);
    }

    /**
     * Create tooltip element
     */
    createTooltip() {
        const tooltip = document.createElement('div');
        tooltip.style.position = 'absolute';
        tooltip.style.padding = '8px 12px';
        tooltip.style.background = 'rgba(0, 0, 0, 0.8)';
        tooltip.style.color = 'white';
        tooltip.style.borderRadius = '4px';
        tooltip.style.pointerEvents = 'none';
        tooltip.style.zIndex = '1000';
        tooltip.style.display = 'none';
        tooltip.style.fontSize = '12px';
        tooltip.style.maxWidth = '200px';
        document.body.appendChild(tooltip);
        return tooltip;
    }

    /**
     * Show tooltip with data
     */
    showTooltip(tooltip, event, data) {
        tooltip.innerHTML = `
            <div><strong>${data.agent || 'Unknown Agent'}</strong></div>
            ${data.model ? `<div>Model: ${data.model}</div>` : ''}
            ${data.benchmark ? `<div>Benchmark: ${data.benchmark}</div>` : ''}
            ${data.accuracy !== undefined ? `<div>Accuracy: ${data.accuracy.toFixed(1)}%</div>` : ''}
            ${data.cost !== undefined ? `<div>Cost: $${data.cost.toFixed(2)}</div>` : ''}
        `;
        tooltip.style.left = event.clientX + 10 + 'px';
        tooltip.style.top = event.clientY - 10 + 'px';
        tooltip.style.display = 'block';
    }

    /**
     * Hide tooltip
     */
    hideTooltip(tooltip) {
        tooltip.style.display = 'none';
    }

    /**
     * Add custom animation function
     */
    addCustomAnimation(animationFunction) {
        this.customAnimations = this.customAnimations || [];
        this.customAnimations.push(animationFunction);
    }

    /**
     * Animation loop
     */
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());

        // Update controls
        if (this.controls) {
            this.controls.update();
        }

        // Run custom animations
        if (this.customAnimations) {
            this.customAnimations.forEach(fn => fn());
        }

        // Render scene
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    /**
     * Cleanup resources
     */
    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.renderer && this.renderer.domElement && this.renderer.domElement.parentNode) {
            this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
            this.renderer.dispose();
        }
        
        // Remove tooltip if exists
        const tooltips = document.querySelectorAll('[style*="position: absolute"][style*="pointer-events: none"]');
        tooltips.forEach(tooltip => tooltip.remove());
    }

    /**
     * Handle window resize
     */
    onWindowResize(width, height) {
        if (this.camera && this.renderer) {
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(width, height);
        }
    }
}

// Export for use in other scripts
window.ThreeJSCharts = ThreeJSCharts;
