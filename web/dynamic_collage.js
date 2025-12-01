import { app } from "../../scripts/app.js";

// ComfyUI Extension for Dynamic Collage Node
// Based on Dapao-Toolbox implementation
app.registerExtension({
    name: "ComfyUI_LLM_Banana.DynamicCollage",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "KenChenLLMGeminiBananaImageCollageNode") {
            console.log("[DynamicCollage] Node registered:", nodeData.name);
            
            // Save original onNodeCreated
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const self = this;
                console.log("[DynamicCollage] Node created");

                // Dynamic input update function
                this.updateInputs = function() {
                    if (!this.inputs) return;

                    // 1. Identify connected inputs and their links
                    const connectedInputs = [];
                    
                    // Scan existing inputs to find connected ones
                    // We need to preserve the link info to re-attach later if needed, 
                    // although standard ComfyUI behavior might handle link preservation if we are careful.
                    // However, strictly "shifting" inputs usually means we are conceptually:
                    // 1. Collecting all valid connected links.
                    // 2. Clearing current inputs (conceptually).
                    // 3. Re-creating inputs sequentially.
                    // 4. Re-attaching links to the new sequential positions.
                    
                    // But ComfyUI API for "moving" links is tricky.
                    // A safer approach for "renaming" behavior without breaking links:
                    // If we detect a gap (e.g., image1=linkA, image2=linkB, image3=empty, image4=linkC),
                    // we want: image1=linkA, image2=linkB, image3=linkC.
                    
                    // Let's gather all currently connected links in order of their appearance
                    const activeLinks = [];
                    
                    for (let i = 0; i < this.inputs.length; i++) {
                        const input = this.inputs[i];
                        const match = input.name.match(/^image(\d+)$/);
                        if (match && input.link !== null) {
                            activeLinks.push({
                                linkId: input.link,
                                type: input.type,
                                originalIndex: i
                            });
                        }
                    }

                    // 2. Determine target structure
                    // We want N connected inputs + 1 empty slot (up to max 14)
                    // Minimum 2 slots (image1, image2) even if 0 connections.
                    
                    let targetCount = activeLinks.length + 1;
                    if (targetCount < 2) targetCount = 2;
                    if (targetCount > 30) targetCount = 30;

                    // 3. Re-organize inputs
                    // This is the hard part. ComfyUI inputs are identified by name in the backend?
                    // Or index? Usually index matches the order in Python's INPUT_TYPES unless dynamic.
                    // But here we are purely frontend.
                    
                    // If we want "image4" to become "image3", we effectively need to:
                    // Option A: Rename the input? (Not standard API)
                    // Option B: Remove the empty "image3" and rename "image4" to "image3"?
                    
                    // Let's try a "Shift Left" strategy.
                    // We iterate through the inputs. If we find an empty input at index `k` (where `k` corresponds to image`k+1`),
                    // and there are connected inputs at `j > k`, we should move the connection from `j` to `k`.
                    // Moving a connection means:
                    // 1. Get the link info from input `j`.
                    // 2. Disconnect input `j`.
                    // 3. Connect that link to input `k`.
                    
                    // However, `app.graph.links` manages the links.
                    // We can use `node.connect()` or similar, but modifying links during an update loop is dangerous.
                    
                    // Better Strategy:
                    // Just remove the empty inputs that are "in the middle".
                    // If we remove an input at index 2 (image3), the input at index 3 (image4) shifts down to index 2.
                    // Does ComfyUI automatically rename it? NO. It will still be named "image4".
                    // So we must also RENAME the remaining inputs to maintain "image1, image2..." naming convention.
                    
                    let changed = false;
                    
                    // Step 3a: Remove empty inputs that are followed by connected inputs (Gaps)
                    // We iterate backwards to safely remove
                    for (let i = this.inputs.length - 2; i >= 0; i--) {
                        const input = this.inputs[i];
                        const match = input.name.match(/^image(\d+)$/);
                        if (!match) continue; // Skip non-image inputs if any
                        
                        if (input.link === null) {
                            // Check if there are any connected inputs AFTER this one
                            let hasFollowingConnection = false;
                            for (let j = i + 1; j < this.inputs.length; j++) {
                                if (this.inputs[j].link !== null && this.inputs[j].name.startsWith("image")) {
                                    hasFollowingConnection = true;
                                    break;
                                }
                            }
                            
                            if (hasFollowingConnection) {
                                console.log(`[DynamicCollage] Removing gap at ${input.name}`);
                                this.removeInput(i);
                                changed = true;
                                // After removal, indices shift, but since we iterate backwards, `i` is still valid for previous items.
                                // But the loop logic for checking "following" needs to be aware? 
                                // Actually, since we restart the logic often via debounce, one removal per cycle is safer?
                                // Let's try to handle one gap at a time or carefully handle indices.
                            }
                        }
                    }

                    // Step 3b: Rename all inputs sequentially to ensure image1, image2, image3...
                    // After removing gaps, we just need to rename what's left.
                    let currentImageIndex = 1;
                    for (let i = 0; i < this.inputs.length; i++) {
                        if (this.inputs[i].name.startsWith("image")) {
                            const newName = `image${currentImageIndex}`;
                            if (this.inputs[i].name !== newName || this.inputs[i].label !== newName) {
                                console.log(`[DynamicCollage] Renaming ${this.inputs[i].name} to ${newName}`);
                                this.inputs[i].name = newName;
                                this.inputs[i].label = newName;
                                changed = true;
                            }
                            currentImageIndex++;
                        }
                    }

                    // Step 3c: Ensure we have the correct number of trailing empty inputs
                    // We want exactly one empty input at the end (up to max 30), but minimum 2 total inputs.
                    
                    // 1. Find the index of the last CONNECTED image input
                    let lastConnectedImageIndex = -1;
                    let imageInputIndices = [];
                    
                    for(let i=0; i<this.inputs.length; i++) {
                         if(this.inputs[i].name.startsWith("image")) {
                             imageInputIndices.push(i);
                             if (this.inputs[i].link !== null) {
                                 lastConnectedImageIndex = imageInputIndices.length - 1; // 0-based index among images
                             }
                         }
                    }

                    // 2. Determine desired number of image inputs
                    // If last connected is index K (e.g. 0 for image1), we want K+2 inputs (image1, image2).
                    // So desired count = lastConnectedImageIndex + 2
                    let desiredImageCount = lastConnectedImageIndex + 2;
                    
                    // Enforce bounds
                    if (desiredImageCount < 2) desiredImageCount = 2;
                    if (desiredImageCount > 30) desiredImageCount = 30;
                    
                    const currentImageCount = imageInputIndices.length;
                    
                    // 3. Add or Remove inputs to match desired count
                    if (currentImageCount < desiredImageCount) {
                        // Add needed inputs
                        for (let i = currentImageCount; i < desiredImageCount; i++) {
                            const nextIndex = i + 1;
                            console.log(`[DynamicCollage] Adding new slot image${nextIndex}`);
                            this.addInput(`image${nextIndex}`, "IMAGE");
                            changed = true;
                        }
                    } else if (currentImageCount > desiredImageCount) {
                        // Remove excess inputs
                        // We remove from the end
                        const removeCount = currentImageCount - desiredImageCount;
                        console.log(`[DynamicCollage] Removing ${removeCount} excess empty slots`);
                        
                        // We need to remove the inputs at specific indices.
                        // Since we remove from end, we can just pop the last ones.
                        // The indices in `this.inputs` correspond to `imageInputIndices` values.
                        // We must remove carefully from end to start to avoid shifting issues affecting subsequent removals in the same loop?
                        // Actually `removeInput` shifts indices of SUBSEQUENT inputs.
                        // So if we remove the last one, the previous ones stay at same index.
                        
                        // Get the real indices to remove
                        const indicesToRemove = imageInputIndices.slice(desiredImageCount);
                        // Reverse them to remove from end
                        indicesToRemove.reverse();
                        
                        for (const realIndex of indicesToRemove) {
                            this.removeInput(realIndex);
                            changed = true;
                        }
                    }
                    
                    if (changed) {
                        // Force redraw if we changed names or inputs
                        this.setDirtyCanvas(true, true);
                    }
                };

                // Listen for connection changes
                const originalOnConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function(type, index, connected, link_info) {
                    // console.log(`[DynamicCollage] Connection changed: type=${type}, index=${index}, connected=${connected}`);
                    if (originalOnConnectionsChange) {
                        originalOnConnectionsChange.apply(this, arguments);
                    }
                    // Debounce update
                    setTimeout(() => {
                        self.updateInputs();
                    }, 20);
                };

                // Initial check
                setTimeout(() => {
                    self.updateInputs();
                }, 100);

                return result;
            };

            // Ensure updates happen after configuration (loading)
            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function() {
                if (originalOnConfigure) {
                    originalOnConfigure.apply(this, arguments);
                }
                setTimeout(() => {
                    this.updateInputs?.();
                }, 100);
            };
        }
    }
});
