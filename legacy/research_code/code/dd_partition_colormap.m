function colors = dd_partition_colormap(nblocks)
%DD_PARTITION_COLORMAP Light blue, white, and neutral grays for DD region maps.
    colors = [
        0.80 0.87 0.93
        0.98 0.98 0.98
        0.78 0.78 0.78
        0.90 0.90 0.90
    ];
    colors = colors(1:nblocks, :);
end
