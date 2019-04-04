--[[
    The implementation of the Iterative Normalization for 4D data (inserted in Convolutional Neural Network) 
    Author: Lei Huang
    Contact: lei.huang@inceptioniai.org
]]--
require('cunn')
local Spatial_DBN_PowerIter,parent = torch.class('nn.Spatial_DBN_PowerIter', 'nn.Module')

function Spatial_DBN_PowerIter:__init(nDim,m_perGroup, nIter,momentum,  affine)
    parent.__init(self)

  --the parameter 'affine' is used to scale the normalized output. If true, scale the output.
    if affine ~= nil then
        assert(type(affine) == 'boolean', 'affine has to be true/false')
        self.affine = affine
    else
        self.affine = false
    end

    if nIter ~= nil then
        self.nIter =nIter 
    else
        self.nIter = 5
    end

--the parameters 'm_perGroup' indicates the number in each group, which is used for group wised whitening
   if m_perGroup~=nil then
      self.m_perGroup = m_perGroup==0 and nDim or m_perGroup>nDim and nDim or m_perGroup 
   else
     self.m_perGroup =  nDim 
   end 

    print('m_perGroup:'.. self.m_perGroup..'----nIter:'.. self.nIter)
    self.train = true
    self.debug = false

    self.eps=1e-5
    self.nDim = nDim
    self.running_means={}  -- the mean used for inference, which is estimated based on each mini-batch with running average
    self.running_projections={}  -- the whitening matrix used for inference, which is also estimated based on each mini-batch with running average
    self.momentum = momentum or 0.1   --running average momentum


     ------------allow nDim % m_perGropu !=0-----
    local groups=torch.floor((nDim-1)/self.m_perGroup)+1
    self.n_groups = groups
 --  print("n_groups:"..self.n_groups)
    for i=1, groups do
        local length = self.m_perGroup
        if i == groups then
            length = nDim - (groups - 1) * self.m_perGroup
        end
        local r_mean=torch.zeros(length)
        local r_projection=torch.eye(length)
        table.insert(self.running_means, r_mean)
        table.insert(self.running_projections, r_projection)
    end
    local length = self.m_perGroup
    self.eye_ngroup = torch.eye(length)

    length = nDim - (groups - 1) * self.m_perGroup
    self.eye_ngroup_last = torch.eye(length)
    if self.affine then
      print('-----------------using scaling-------------------')
        self.weight = torch.Tensor(nDim)
       self.bias = torch.Tensor(nDim)
       self.gradWeight = torch.Tensor(nDim)
       self.gradBias = torch.Tensor(nDim)
       self:reset()
    end

end

function Spatial_DBN_PowerIter:reset()
   self.weight:fill(1)
   self.bias:zero()
end

function Spatial_DBN_PowerIter:updateOutput(input)
   assert(input:dim() == 4, 'only mini-batch supported (4D tensor), got '
             .. input:dim() .. 'D tensor instead')
----------------------------- train mode---------------------
  function update_perGroup(data,groupId)
     local scale=data.new() --eigValue^(-1/2)
     local temp_1D= data.new()
     local centered = data.new()
     local output = data.new()
     local sigma = data.new()
     local set_X = {}
     
     local nBatch=data:size(1)
     local nFeature=data:size(2)
     local eye_nDim
     if groupId ~= self.n_groups then 
       eye_nDim=self.eye_ngroup 
     else
       eye_nDim=self.eye_ngroup_last 
     end 
     
     centered:resizeAs(data)
     output:resizeAs(data)

     temp_1D:mean(data, 1)  -- E(x) = expectation of x.
     self.running_means[groupId]:mul(1 - self.momentum):add(self.momentum, temp_1D)

      -- subtract mean
     centered:copy(data):add(-temp_1D:expand(nBatch,nFeature))

      ----------------------calcualte the projection matrix----------------------
     sigma:resize(nFeature,nFeature)
     sigma:addmm(0,sigma,1/nBatch,centered:t(),centered)  --buffer_1 record correlation matrix
     sigma:add(self.eps, eye_nDim)      
     local trace=torch.trace(sigma)
     local sigma_norm=sigma/trace 
     local X=eye_nDim:clone() 
     for i=1,self.nIter do
        X=(3*X-X*X*X*sigma_norm)/2
        table.insert(set_X, X:clone())
      -- print(X)
     end
     
     local whiten_matrix=X/torch.sqrt(trace)
      self.running_projections[groupId]:mul(1 - self.momentum):add(self.momentum, whiten_matrix) -- add to running projection
     output:mm(centered, whiten_matrix)
      
      if self.debug then            
        print(self.running_means) 
        print(self.running_projections) 
      end
      ----------------record the results of per group--------------
      table.insert(self.centereds, centered)
      table.insert(self.sigmas, sigma)
      table.insert(self.whiten_matrixs, whiten_matrix)
      table.insert(self.set_Xs, set_X)
   --  print(self.set_Xs) 
      return output
 end
 
 
  ------------------------ test mode---------------------
 
  function test_perGroup(data,groupId)
    local nBatch = data:size(1)
    local nFeature=data:size(2)
    self.buffer_test=self.buffer_test or data.new()
    --local output=data.new()
    self.buffer_test:resizeAs(data):copy(data)
    self.buffer_test:add(-1,self.running_means[groupId]:view(1,nFeature):expandAs(data))

    self.buffer_test:mm(self.buffer_test,self.running_projections[groupId])
    return self.buffer_test
  end
 
---------------------------------------------------------------------------------------
-------------------updateGradInput main function-------------------------
----------------------------------------------------------------------------------------
   local nBatch = input:size(1)
   local nDim=input:size(2)
   local iH=input:size(3)
   local iW=input:size(4)
   self.sigmas={}
   self.set_Xs={}
   self.centereds={}
   self.whiten_matrixs={}
   
   local groups=torch.floor((nDim-1)/self.m_perGroup)+1
    
   self.output=self.output or input.new()
   self.output:resize(#input)
   self.normalized = self.normalized or input.new()
   self.normalized:resizeAs(input)

   self.gradInput=self.gradInput or input.new()
   self.gradInput:resize(#input)
  
   self.buffer = self.buffer or input.new()
   self.input_temp= self.input_temp or input.new() 
   self.output_temp= self.output_temp or input.new() 
   self.input_temp=input:view(nBatch,nDim,iH*iW):transpose(1,2):reshape(nDim,nBatch*iH*iW):t()  --transfer to 2D data
   self.output_temp:resizeAs(self.input_temp)
   
   if self.train == false then
      for i=1,groups do 
        local start_index=(i-1)*self.m_perGroup+1
        local end_index=math.min(i*self.m_perGroup,nDim)      
        self.output_temp[{{},{start_index,end_index}}]=test_perGroup(self.input_temp[{{},{start_index,end_index}}],i)   
      end
   else -- training mode
   -- print('-------------------------training mode-----------------') 
      for i=1,groups do 
        local start_index=(i-1)*self.m_perGroup+1
        local end_index=math.min(i*self.m_perGroup,nDim)    
        self.output_temp[{{},{start_index,end_index}}]=update_perGroup(self.input_temp[{{},{start_index,end_index}}],i)   
      end
   end
     self.output:copy(self.output_temp:t():reshape(nDim, nBatch,iH*iW):transpose(1,2):reshape(nBatch,nDim,iH,iW)) 
     self.normalized:copy(self.output)


     if self.affine then
        self.output:cmul(self.weight:view(1,nDim,1,1):expandAs(self.output))
        self.output:add(self.bias:view(1,nDim,1,1):expandAs(self.output))
     end
    collectgarbage()
    return self.output
end

function Spatial_DBN_PowerIter:updateGradInput(input, gradOutput)

   function Matrix_Pow3(input)
     local b=torch.mm(input, input)
     return torch.mm(b,input) 
   end

   function updateGradInput_perGroup(gradOutput_perGroup,groupId)
     local  sigma=self.sigmas[groupId]
     local  centered=self.centereds[groupId]
     local  whiten_matrix=self.whiten_matrixs[groupId]
     local  set_X=self.set_Xs[groupId]
     local nBatch = gradOutput_perGroup:size(1) 
     local nFeature = gradOutput_perGroup:size(2) 
     local trace=torch.trace(sigma)
     local sigma_norm=sigma/trace
     local eye_nDim
     self.dC=self.dC or gradOutput_perGroup.new()
     self.dA=self.dA or gradOutput_perGroup.new()
     self.dSigma=self.dSigma or gradOutput_perGroup.new()
     self.dXN=self.dXN or gradOutput_perGroup.new()
     self.f=self.f or gradOutput_perGroup.new()
     self.dC:resizeAs(whiten_matrix) 
     self.dA:resizeAs(whiten_matrix) 
     
     self.dC:mm(gradOutput_perGroup:t(), centered)
     self.dXN=self.dC/torch.sqrt(trace) 
     if groupId ~= self.n_groups then 
       eye_nDim=self.eye_ngroup 
     else
       eye_nDim=self.eye_ngroup_last 
     end 
--    print(#set_X)    
     local P3 
     if self.nIter==1 or self.nIter==0 then 
         P3=eye_nDim
     else
       P3=Matrix_Pow3(set_X[self.nIter-1])
     end
     --print(P3)
     self.dA:mm(P3:t(),self.dXN)
     local dX_kPlus=self.dXN
     for i=self.nIter-1, 1, -1 do
         ----calculate dL/dx_k+1--------------
       local X=set_X[i]
       local tmp1=dX_kPlus*(X*X*sigma_norm):t()
       local tmp2=X:t()* dX_kPlus *(X*sigma_norm):t()
       local tmp3=(X*X):t()* dX_kPlus * sigma_norm:t() 
       local dX_k= 3*dX_kPlus/2-(tmp1+tmp2+tmp3)/2 
       
      ------update dA-------------- 
       if  i ~= 1  then
         local    X_before=set_X[i-1]
         local tmp=Matrix_Pow3(X_before)
         self.dA:add(tmp:t()*dX_k)
         dX_kPlus=dX_k
       else 
         self.dA:add(dX_k) 
       end
     end
    
     self.dA=-self.dA/2

     if self.nIter==0 then
        self.dSigma=-torch.trace(self.dC)/(2*trace^(3/2)) * eye_nDim
     else
        local s1=torch.trace(self.dA:t()*sigma)
        local s2=torch.trace(self.dC:t()*set_X[self.nIter]) 
        self.dSigma=-(s1/(trace^2)+s2/(2*trace^(3/2)))*eye_nDim
        self.dSigma=self.dSigma+self.dA/trace
     end
     
     local dSigma_sym=(self.dSigma+self.dSigma:t())/2

     self.f:mean(gradOutput_perGroup, 1)
     local d_mean=gradOutput_perGroup-self.f:expandAs(gradOutput_perGroup) 
     self.buffer:resizeAs(gradOutput_perGroup) 
     self.buffer:mm(d_mean, whiten_matrix) 
     local gradInput=(2/nBatch)*centered*dSigma_sym
     gradInput=gradInput+self.buffer  
    
     return gradInput
  end
  

----------------------------------------------------------------  
 -------update the gradInput per Group in test mode:  this mode may be benifit based on the Batch Renomrlaizaiton methods----------------------------------------------------- 
  function updateGradInput_perGroup_test(gradOutput_perGroup,groupId)
     local running_projection=self.running_projections[groupId] --use running projection
     local gradInput=gradOutput_perGroup.new()
     gradInput:resizeAs(gradOutput_perGroup)
     gradInput:mm(gradOutput_perGroup,running_projection:t())
     return gradInput
  end
---------------------------------------------------------------------------------------
-------------------updateGradInput main function-------------------------
----------------------------------------------------------------------------------------


   assert(input:dim() == 4, 'only mini-batch supported')
   assert(gradOutput:dim() == 4, 'only mini-batch supported')
   local nBatch=input:size(1)
   local nDim=input:size(2)
   local iH=input:size(3)
   local iW=input:size(4)
   local groups=torch.floor((nDim-1)/self.m_perGroup)+1

   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   if self.affine then
     self.gradInput:cmul(self.weight:view(1,nDim,1,1):expandAs(self.gradInput))
   end

   self.input_temp=self.gradInput:view(nBatch,nDim,iH*iW):transpose(1,2):reshape(nDim,nBatch*iH*iW):t()  --transfoer to 2D data
   self.output_temp:resizeAs(self.input_temp)
  
   if self.train==false then 
      for i=1,groups do 
         local start_index=(i-1)*self.m_perGroup+1
         local end_index=math.min(i*self.m_perGroup,nDim)   
         self.output_temp[{{},{start_index,end_index}}]=updateGradInput_perGroup_test(self.input_temp[{{},{start_index,end_index}}],i)   
      end
   else
     for i=1,groups do 
         local start_index=(i-1)*self.m_perGroup+1
         local end_index=math.min(i*self.m_perGroup,nDim)   
         self.output_temp[{{},{start_index,end_index}}]=updateGradInput_perGroup(self.input_temp[{{},{start_index,end_index}}],i)   
     end
   end 
   self.gradInput:copy(self.output_temp:t():reshape(nDim, nBatch,iH*iW):transpose(1,2):reshape(nBatch,nDim,iH,iW)) 
     
   collectgarbage()
   return self.gradInput
end

function Spatial_DBN_PowerIter:setTrainMode(isTrain)
  if isTrain ~= nil then
      assert(type(isTrain) == 'boolean', 'isTrain has to be true/false')
      self.train = isTrain
  else
    self.train=true  

  end
end

function Spatial_DBN_PowerIter:accGradParameters(input, gradOutput, scale)
  if self.affine then
    scale = scale or 1.0
    local nBatch = input:size(1)
    local nFeature = input:size(2)
    local iH = input:size(3)
    local iW = input:size(4)
    self.output_temp:resizeAs(self.normalized):copy(self.normalized)
    self.output_temp = self.output_temp:cmul(gradOutput):view(nBatch, nFeature, iH*iW)
    self.buffer:sum(self.output_temp, 1) -- sum over mini-batch
    self.output_temp:sum(self.buffer, 3) -- sum over pixels
    self.gradWeight:add(scale, self.output_temp)
    self.buffer:sum(gradOutput:view(nBatch, nFeature, iH*iW), 1)
    self.output_temp:sum(self.buffer, 3)
    self.gradBias:add(scale, self.output_temp) -- sum over mini-batch
  end
end
function Spatial_DBN_PowerIter:clearState()
  print('-----------clear the memory--------------------')
       -- self.buffer:set()
       -- self.output_temp:set()
      --   self.input_temp:set()
      --   self.gradInput:set()
      --   self.dC:set()
      --   self.dSigma:set()
      --   self.dXN:set()
      --   self.f:set()
      --  self.eye_ngroup:set()
      --  self.eye_ngroup_last:set()
      --  self.output:set()
  return 
end
