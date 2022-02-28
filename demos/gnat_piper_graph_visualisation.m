%% This script constructs the graphs from
% A population-based SHM methodology for heterogeneous structures:
% transferring damage localisation knowledge between different aircraft
% wings
%
% Paul Gardner, University of Sheffield 2022
%
% Note when boundary node is included in the graph the boundary node is
% numbered 1 and all other nodes are shifted up by one

clear all
close all
clc

%% Gnat into Piper Tomahawk panel combinations

% find all graphs - in this case collection of nodes that form the subgraph
C = (factorial(9)/(factorial(5)*factorial(9-5))); % binomial coefficent
Cp = factorial(5); % no. of permutations
no_combinations = C*Cp; % binomial coefficent * permutations

nchoosek_combinations = nchoosek(1:9,5); % all combinations without permutations
combs_all = reshape(nchoosek_combinations(:,fliplr(perms(1:5))),[],5); % all combinations with permutations

%% Graphs with boundary nodes

% Gnat graph - boundary nodes
EdgeTable = table([1,2; 1,3; 1,4; 1,5; 2,7; 2,3; 3,7; 3,4; 4,7; 4,5; 5,6; 6,7; 7,10; 6,10; 10,9; 9,8],'VariableNames',{'EndNodes'});
NodeTable = table({'F';'1';'2';'3';'4';'5';'6';'7';'8';'9'},'VariableNames',{'EndNodes'});
G_gnat = graph(EdgeTable,NodeTable);

% Piper Tomahawk graph - boundary nodes
EdgeTable = table([1,2; 2,3; 3,4; 4,5; 5,6],'VariableNames',{'EndNodes'});
NodeTable = table({'M';'1';'2';'3';'4';'5'},'VariableNames',{'EndNodes'});
G_piper = graph(EdgeTable,NodeTable);

figure
subplot(1,2,1)
plot(G_gnat);
xlabel('Gnat with boundary node')
subplot(1,2,2)
plot(G_piper);
xlabel('Piper with boundary node')

%% Combinations that include boundary node

% find all combinations of paths that connect 6 nodes (i.e. paths of 5
% edges starting at the boundary node)
start_finish = [ones(9,1), (2:10)']; % all start and end values
k = 1;
paths6 = []; % matrix of nodes in path
edges6 = []; % matrix of path edges
for i = 1:length(start_finish)
    
    [paths, edges] = allpaths(G_gnat,start_finish(i,1),start_finish(i,2),...
        'MinPathLength',5,'MaxPathLength',5); % find all paths between
    % start and finish of length 5 (i.e. 5 edges)
    
    if ~isempty(paths) % path of length 4 exists append to matrix
        pl = size(paths,1);
        paths6(k:k+(pl-1),:) = cell2mat(paths);
        edges6(k:k+(pl-1),:) = cell2mat(edges);
        k = k + pl;
    end
end
paths6 = paths6(:,2:end)-1;

%% Graphs without boundary nodes

% Gnat graph - no boundary nodes
EdgeTable = table([1,6; 1,2; 2,6; 2,3; 3,6; 3,4; 4,5; 5,6; 6,9; 5,9; 9,8; 8,7],'VariableNames',{'EndNodes'});
NodeTable = table({'1';'2';'3';'4';'5';'6';'7';'8';'9'},'VariableNames',{'EndNodes'});
G_gnat_nb = graph(EdgeTable,NodeTable);


% Piper Tomahawk graph - boundary nodes
EdgeTable = table([1,2; 2,3; 3,4; 4,5],'VariableNames',{'EndNodes'});
NodeTable = table({'1';'2';'3';'4';'5'},'VariableNames',{'EndNodes'});
G_piper_nb = graph(EdgeTable,NodeTable);

figure
subplot(1,2,1)
plot(G_gnat_nb);
xlabel('Gnat without boundary node')
subplot(1,2,2)
plot(G_piper_nb);
xlabel('Piper without boundary node')

%% Combinations that do not include boundary node

% find all combinations of paths that connect 5 nodes (i.e. paths of 4
% edges)
start_finish = nchoosek(1:9,2); % all start and end values
k = 1;
paths5 = []; % matrix of nodes in path
edges5 = []; % matrix of path edges
for i = 1:length(start_finish)
    
    [paths, edges] = allpaths(G_gnat_nb,start_finish(i,1),start_finish(i,2),...
        'MinPathLength',4,'MaxPathLength',4); % find all paths between
    % start and finish of length 4 (i.e. 4 edges)
    
    if ~isempty(paths) % path of length 4 exists append to matrix
        pl = size(paths,1);
        paths5(k:k+(pl-1),:) = cell2mat(paths);
        edges5(k:k+(pl-1),:) = cell2mat(edges);
        k = k + pl;
    end
end

%% Maximum common subgraphs located from the modified BK algorithm

% maximum common subgraphs - from graph matching
maxsub(1,:) = [4 5 9 8 7]; % (a)
maxsub(2,:) = [1 6 9 8 7]; % (b)
maxsub(3,:) = [2 6 9 8 7]; % (c)
maxsub(4,:) = [3 6 9 8 7]; % (d)

%% Graph labels
% label graph space
% first column is what permutation in 9 choose 5
% second column:
% 0: unstructured (random), 1: path on graph without boundary node,
% 2: path on graph with boundary node, 3: maximum common subgraphs

Y_graph = zeros(size(combs_all,1),2); % unstructured graphs (zero label)

Y_graph(:,1) = repmat(1:C,1,Cp)'; % label of what permutation in 9 choose 5
[~,ind] = sort(Y_graph(:,1)); % sort label list
Y_graph = Y_graph(ind,:);
combs_all = combs_all(ind,:); % sort all graphs by permutation order

% a path in graph without boundary node (one label)
edge_ind = nan(size(edges5,1),1);
for i = 1:size(paths5,1)
    Y_graph(all(combs_all==paths5(i,:),2),2) = 1;
    edge_ind(i) = find(all(combs_all==paths5(i,:),2));
end

% a path in graph with boundary node (two label)
edge_ind6 = nan(size(edges6,1),1);
for i = 1:size(paths6,1)
    Y_graph(all(combs_all==paths6(i,:),2),2) = 2;
    edge_ind6(i) = find(all(combs_all==paths6(i,:),2));
end

% maximum common subgraph (three label)
for i = 1:size(maxsub,1)
    Y_graph(all(combs_all==maxsub(i,:),2),2) = 3;
end

%% Visual check - plot graphs with valid paths

figure
for i = 1:no_combinations
    if Y_graph(i,2) == 1
        subplot(1,2,1)
        p = plot(G_gnat_nb);
        highlight(p,combs_all(i,:),'edges',edges5(find(edge_ind==i,1,'first'),:),'EdgeColor','r','LineWidth',1.5,'NodeColor','r','MarkerSize',6);
        xlabel('Gnat without boundary node')
        subplot(1,2,2)
        pp = plot(G_piper_nb);
        highlight(pp,1:5,'edges',[1; 2; 3; 4],'EdgeColor','r','LineWidth',1.5,'NodeColor','r','MarkerSize',6);
        xlabel('Piper without boundary node')
        pause(2)
    elseif Y_graph(i,2) == 2
        subplot(1,2,1)
        p = plot(G_gnat);
        highlight(p,[1,combs_all(i,:)+1],'edges',edges6(find(edge_ind6==i,1,'first'),:),'EdgeColor','m','LineWidth',1.5,'NodeColor','m','MarkerSize',6);
        xlabel('Path with boundary node')
        subplot(1,2,2)
        pp = plot(G_piper);
        highlight(pp,1:6,'edges',[1; 2; 3; 4; 5],'EdgeColor','m','LineWidth',1.5,'NodeColor','m','MarkerSize',6);
        xlabel('Piper with boundary node')
        pause(2)
    end
end

%% Visual check maximum common subgraphs

edges = [4, 11, 13, 15, 16;...
    1, 6, 14, 15, 16;...
    2, 8, 14, 15, 16;...
    3, 10, 14, 15, 16];

mcs = {'(a)','(b)','(c)','(d)'};

figure('position',[500,500,1000,300])
for i = 1:size(maxsub,1)
    subplot(1,5,i)
    p = plot(G_gnat);
    highlight(p,[1,maxsub(i,:)+1],'edges',edges(i,:),'EdgeColor','k','LineWidth',1.5,'NodeColor','k','MarkerSize',6);
    xlabel(mcs{i})
end
subplot(1,5,5)
pp = plot(G_piper);
highlight(pp,1:6,'edges',[1; 2; 3; 4; 5],'EdgeColor','k','LineWidth',1.5,'NodeColor','k','MarkerSize',6);
xlabel('Piper')