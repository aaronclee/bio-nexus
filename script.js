// script.js

/**
 * This script loads and visualizes a comprehensive undirected knowledge graph.
 * Enhancements include:
 * 1. Revised search functionality to trigger on pressing the Enter key.
 * 2. Clearer user feedback when no matches are found.
 * 3. Nodes and edges are highlighted correctly based on the query.
 * 4. All existing functionalities, including overlays and color coding, remain intact.
 * 5. Comprehensive code documentation and clarity.
 */

d3.json("data/knowledge_graph.json").then(data => {  // change to test_graph.json for unit tests
    // Process nodes and edges from JSON data
    const nodes = Object.keys(data.nodes).map(key => ({
        id: key,
        ...data.nodes[key].properties
    }));

    // Convert edges to a suitable format for D3
    const edges = Object.keys(data.edges).map(key => {
        const edge = data.edges[key];
        return {
            ...edge,
            source: edge.source_node,
            target: edge.target_node
        };
    });

    // Dictionaries for fast node lookups
    const nodeById = Object.fromEntries(nodes.map(n => [n.id, n]));
    edges.forEach(e => {
        // Convert source and target references from string IDs to node objects
        if (typeof e.source === "string") {
            e.source = nodeById[e.source];
        }
        if (typeof e.target === "string") {
            e.target = nodeById[e.target];
        }
    });

    // Prepare distinct node and edge types for legends, filters, and coloring
    const nodeTypes = Array.from(new Set(nodes.map(d => d.entity_type))).sort();
    const edgeTypes = Array.from(new Set(edges.map(d => d.relationship_type))).sort();

    // Define color scales for nodes and edges
    const nodeColor = d3.scaleOrdinal()
        .domain(nodeTypes)
        .range(d3.schemeCategory10);

    const edgeColor = d3.scaleOrdinal()
        .domain(edgeTypes)
        .range(d3.schemeCategory10);  // Color scheme for edges

    // Dimensions
    const graphContainer = document.getElementById('graph');
    const width = graphContainer.clientWidth;
    const height = graphContainer.clientHeight;

    // Creating an SVG element to hold the graph
    const svg = d3.select("#graph")
        .append("svg")
        .attr("width", "100%")
        .attr("height", "100%");

    // Group to apply transformations (zooming/panning) to nodes and edges
    const g = svg.append("g")
        .attr("class", "all");

    // Zoom behavior for the graph
    const zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on("zoom", (event) => {
            g.attr("transform", event.transform);
        });

    svg.call(zoom);

    // Force simulation for positioning nodes and edges
    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(edges).id(d => d.id).distance(200))
        .force("charge", d3.forceManyBody().strength(-200))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide().radius(30).iterations(2));

    // Adding edges to graph
    const link = g.append("g")
        .attr("class", "edges")
        .selectAll("line")
        .data(edges)
        .enter()
        .append("line")
        .attr("class", "edge")
        .attr("stroke", d => edgeColor(d.relationship_type))
        .attr("stroke-width", 4)
        .attr("opacity", 0.8)
        .on("mouseover", showEdgeTooltip)
        .on("mouseout", hideTooltip)
        .on("click", showEdgeDetails);

    // Adding nodes to graph
    const node = g.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(nodes)
        .enter()
        .append("circle")
        .attr("class", "node")
        .attr("r", 10)
        .attr("fill", d => nodeColor(d.entity_type))
        .on("mouseover", highlightConnectedEdges)
        .on("mouseout", resetEdgeHighlight)
        .on("click", showNodeDetails)
        .call(d3.drag()
            .on("start", dragStarted)
            .on("drag", dragged)
            .on("end", dragEnded));

    // Adding labels to nodes for fast ID
    const labels = g.append("g")
        .attr("class", "labels")
        .selectAll("text.node-label")
        .data(nodes)
        .enter()
        .append("text")
        .attr("class", "node-label")
        .attr("dy", -15)
        .attr("text-anchor", "middle")
        .text(d => d.primary_name);

    // Creating a tooltip for hover interactions
    const tooltip = d3.select("body")
        .append("div")
        .attr("class", "tooltip");

    /** -------------- Functions to handle user interactions -------------- **/

    // Zoom control buttons
    d3.select("#zoom_in").on("click", () => {
        zoom.scaleBy(svg.transition().duration(500), 1.2);
    });

    d3.select("#zoom_out").on("click", () => {
        zoom.scaleBy(svg.transition().duration(500), 0.8);
    });

    d3.select("#reset_view").on("click", () => {
        svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
    });

    // **Enhanced Search Functionality**:
    // Now the search is triggered when the user presses the Enter key.
    d3.select("#search").on("keydown", function(event) {
        if (event.key === 'Enter') {
            const query = this.value.toLowerCase().trim();
            resetHighlights(); // Reset highlights before applying new search results

            if (query) {
                // Matching nodes by name
                const matchedNodes = nodes.filter(node => node.primary_name.toLowerCase().includes(query));
                // Creating a set of matched node IDs for easier membership testing
                const matchedNodeIds = new Set(matchedNodes.map(n => n.id));

                // Matching edges by relationship type or connecting matched nodes
                const matchedEdges = edges.filter(edge => 
                    edge.relationship_type.toLowerCase().includes(query) ||
                    matchedNodeIds.has(edge.source.id) ||
                    matchedNodeIds.has(edge.target.id)
                );

                // Highlight matched nodes
                node.attr("stroke", d => matchedNodeIds.has(d.id) ? "#FF0000" : null)
                    .attr("stroke-width", d => matchedNodeIds.has(d.id) ? 3 : 1);

                // Highlight matched edges
                link.attr("stroke", d => matchedEdges.includes(d) ? "#FF0000" : edgeColor(d.relationship_type))
                    .attr("stroke-width", d => matchedEdges.includes(d) ? 6 : 4);

                // If no matches found, provide feedback
                if (matchedNodes.length === 0 && matchedEdges.length === 0) {
                    d3.select("#details").html("<strong>No matches found for your search query.</strong>");
                } else {
                    d3.select("#details").html("<strong>Search results highlighted. Use filters or hover for details.</strong>");
                }
            } else {
                // If query is empty, reset highlights and details
                d3.select("#details").html("Click on a node or edge to see details");
            }
        }
    });

    // On each tick of the simulation, update node and edge positions
    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);

        labels
            .attr("x", d => d.x)
            .attr("y", d => d.y - 15);
    });

    // Display node details in the details panel
    function showNodeDetails(event, d) {
        d3.select("#details").html(generateNodeDetailsHTML(d));
    }

    // Display edge details in the details panel
    function showEdgeDetails(event, d) {
        d3.select("#details").html(generateEdgeDetailsHTML(d));
    }

    // Node hover highlight connected edges
    function highlightConnectedEdges(event, d) {
        // Highlight edges connected to this node
        link.attr("stroke", edgeData => 
            edgeData.source.id === d.id || edgeData.target.id === d.id ? "#FF0000" : edgeColor(edgeData.relationship_type)
        ).attr("stroke-width", edgeData => 
            edgeData.source.id === d.id || edgeData.target.id === d.id ? 6 : 4
        );
    }

    // Reset edge highlight when mouse leaves the node
    function resetEdgeHighlight() {
        link.attr("stroke", d => edgeColor(d.relationship_type)).attr("stroke-width", 4);
    }

    // Show node tooltip on hover
    function showNodeTooltip(event, d) {
        tooltip.style("display", "inline");
        tooltip.html(`<strong>${d.primary_name}</strong><br>Type: ${d.entity_type}`)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY + 10) + "px");
    }

    // Show edge tooltip on hover
    function showEdgeTooltip(event, d) {
        tooltip.style("display", "inline");
        tooltip.html(`<strong>Relationship:</strong> ${d.relationship_type}`)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY + 10) + "px");
    }

    // Hide tooltip on mouse out
    function hideTooltip() {
        tooltip.style("display", "none");
    }

    // Drag event handlers for node movement
    function dragStarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragEnded(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    /** -------------- Functions to handle Filters and Highlights -------------- **/

    // Node type filters
    const nodeFiltersList = d3.select("#node-filters-list");
    nodeTypes.forEach(type => {
        const listItem = nodeFiltersList.append("li");
        const checkbox = listItem.append("input")
            .attr("type", "checkbox")
            .attr("checked", true)
            .attr("value", type);
        listItem.append("span").text(` ${type}`);

        checkbox.on("change", function() {
            toggleNodeVisibility(this.value, this.checked);
        });
    });

    // Edge type filters
    const edgeFiltersList = d3.select("#edge-filters-list");
    edgeTypes.forEach(type => {
        const listItem = edgeFiltersList.append("li");
        const checkbox = listItem.append("input")
            .attr("type", "checkbox")
            .attr("checked", true)
            .attr("value", type);
        listItem.append("span").text(` ${type}`);

        checkbox.on("change", function() {
            toggleEdgeVisibility(this.value, this.checked);
        });
    });

    // Toggle node visibility based on the node's entity_type
    function toggleNodeVisibility(type, visible) {
        node.each(function(d) {
            if (d.entity_type === type) {
                d3.select(this).classed("hidden", !visible);
                // Hide or show connected edges accordingly
                link.filter(e => e.source.id === d.id || e.target.id === d.id)
                    .classed("hidden", !visible);
            }
        });
        labels.classed("hidden", d => !visible && d.entity_type === type);
    }

    // Toggle edge visibility based on the edge's relationship_type
    function toggleEdgeVisibility(type, visible) {
        link.filter(d => d.relationship_type === type)
            .classed("hidden", !visible);
    }

    // Reset all highlights (for both nodes and edges)
    function resetHighlights() {
        node.attr("stroke", null).attr("stroke-width", 1);
        link.attr("stroke", d => edgeColor(d.relationship_type)).attr("stroke-width", 4);
        d3.select("#details").html("Click on a node or edge to see details");
    }

    /** -------------- Functions to generate HTML for details panel -------------- **/

    // Generate HTML for node details using all node properties
    function generateNodeDetailsHTML(nodeData) {
        return `
            <strong>Node Details:</strong><br>
            <table>
                <tr><td><strong>Type:</strong></td><td>${nodeData.entity_type}</td></tr>
                <tr><td><strong>Name:</strong></td><td>${nodeData.primary_name}</td></tr>
                <tr><td><strong>Description:</strong></td><td>${nodeData.description}</td></tr>
                <tr><td><strong>Alternative Names:</strong></td><td>${nodeData.alternative_names.join(", ") || "N/A"}</td></tr>
                <tr><td><strong>External IDs:</strong></td><td>${formatExternalIds(nodeData.external_ids)}</td></tr>
                <tr><td><strong>Last Updated:</strong></td><td>${nodeData.last_updated}</td></tr>
                <tr><td><strong>Creation Date:</strong></td><td>${nodeData.creation_date}</td></tr>
            </table>
        `;
    }

    // Generate HTML for edge details using all edge properties including evidence and metadata
    function generateEdgeDetailsHTML(edgeData) {
        return `
            <strong>Edge Details:</strong><br>
            <table>
                <tr><td><strong>Relationship:</strong></td><td>${edgeData.relationship_type}</td></tr>
                <tr><td><strong>Source Node:</strong></td><td>${edgeData.source.primary_name}</td></tr>
                <tr><td><strong>Target Node:</strong></td><td>${edgeData.target.primary_name}</td></tr>
                ${formatEvidence(edgeData.evidence)}
                ${formatAggregatedMetadata(edgeData.aggregated_metadata)}
            </table>
        `;
    }

    /** -------------- Helper functions for formatting complex data -------------- **/

    // Format external IDs if present
    function formatExternalIds(externalIds) {
        if (!externalIds || Object.keys(externalIds).length === 0) {
            return "N/A";
        }
        return Object.entries(externalIds).map(([key, value]) => `${key}: ${value}`).join(", ");
    }

    // Format edge evidence details 
    function formatEvidence(evidenceArray) {
        if (!evidenceArray || evidenceArray.length === 0) {
            return `<tr><td><strong>Evidence:</strong></td><td>N/A</td></tr>`;
        }

        return evidenceArray.map(e => `
            <tr>
                <td><strong>Evidence (Paper ID: ${e.paper_id}):</strong></td>
                <td>
                    <p><strong>Title:</strong> ${e.citation_metadata.title}</p>
                    <p><strong>Authors:</strong> ${e.citation_metadata.authors.join(", ")}</p>
                    <p><strong>Journal:</strong> ${e.citation_metadata.journal}</p>
                    <p><strong>Year:</strong> ${e.citation_metadata.year}</p>
                    <p><strong>Extracted Text:</strong> ${e.extracted_text}</p>
                </td>
            </tr>
        `).join("");
    }

    // Format aggregated metadata if present
    function formatAggregatedMetadata(metadata) {
        if (!metadata) {
            return `<tr><td><strong>Aggregated Metadata:</strong></td><td>N/A</td></tr>`;
        }

        return `
            <tr><td colspan="2"><strong>Aggregated Metadata:</strong></td></tr>
            <tr><td>Total Papers:</td><td>${metadata.total_papers}</td></tr>
            <tr><td>Earliest Evidence:</td><td>${metadata.earliest_evidence}</td></tr>
            <tr><td>Latest Evidence:</td><td>${metadata.latest_evidence}</td></tr>
            <tr><td>Evidence Strength:</td><td>${metadata.evidence_strength}</td></tr>
            <tr><td>Contradictory Evidence:</td><td>${metadata.contradictory_evidence}</td></tr>
            <tr><td>Last Updated:</td><td>${metadata.last_updated}</td></tr>
        `;
    }

    // Create legends for node and edge types
    const nodeLegend = d3.select("#node-legend");
    nodeTypes.forEach(t => {
        nodeLegend.append("li")
            .html(`<span class="legend-color-box" style="background:${nodeColor(t)}"></span>${t}`);
    });

    const edgeLegend = d3.select("#edge-legend");
    edgeTypes.forEach(t => {
        edgeLegend.append("li")
            .html(`<span class="legend-color-box" style="background:${edgeColor(t)}"></span>${t}`);
    });
}).catch(error => console.error(error));
