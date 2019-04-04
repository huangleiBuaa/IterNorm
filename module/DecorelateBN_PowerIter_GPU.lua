--[[
    The implementation of the Iterative Normalization for 2D data (inserted in MLP or full connected layers) 
    Author: Lei Huang
    Contact: lei.huang@inceptioniai.org
]]--
local DecorelateBN_PowerIter_GPU,parent = torch.class('nn.DecorelateBN_PowerIter_GPU', 'nn.Module')

function DecorelateBN_PowerIter_GPU:__init(nDim, m_perGroup, nIter, momentum,affine)
   parent.__init(self)
   
   if affine ~= nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = false
   end

   if m_perGroup~=nil then
      self.m_perGroup = m_perGroup==0 and nDim or m_perGroup>nDim and nDim or m_perGroup 
   else
     self.m_perGroup =  nDim 
   end 
   if nIter~=nil then
      self.nIter = nIter 
   else
     self.nIter =5 
   end 
   print('m_perGroup:'.. self.m_perGroup..'----nIter:'.. self.nIter)

   self.nDim=nDim  
   self.momentum = momentum or 0.1
   self.running_means={}
   self.running_projections={}
 
   local groups=torch.floor((nDim-1)/self.m_perGroup)+1
   self.n_groups=groups
   ------------allow nDim % m_perGropu !=0-----
   for i=1,groups do
       local length=self.m_perGroup   
       if i==groups then
          length = nDim - (groups - 1) * self.m_perGroup
       end
       local r_mean=torch.zeros(length)
        local r_projection=torch.eye(length)
        table.insert(self.running_means, r_mean)
        table.insert(self.running_projections,r_projection)
   end

   local length = self.m_perGroup
    self.eye_ngroup = torch.eye(length)
    length = nDim - (groups - 1) * self.m_perGroup
    self.eye_ngroup_last = torch.eye(length)
   ----do axis-wise scaling along the projected orthogonal directions-----
   if self.affine then
    print('---------------------------using scale-----------------')
       self.weight = torch.Tensor(nDim)
      self.bias = torch.Tensor(nDim)
      self.gradWeight = torch.Tensor(nDim)
      self.gradBias = torch.Tensor(nDim)
      self.flag_inner_lr=false
      self.scale=1 --which is used to scale the affine_weight
      self:reset()
   end
   
 -----some configures------------
   self.debug=false
   self.train =true   
end


function DecorelateBN_PowerIter_GPU:reset()
  -- self.weight:uniform()
   self.weight:fill(1)
   self.bias:zero()
end


function DecorelateBN_PowerIter_GPU:updateOutput(input)
   assert(input:dim() == 2, 'only mini-batch supported (2D tensor), got '
             .. input:dim() .. 'D tensor instead')
             
------------------------------------------train mode -------------------------------
 
  function updateOutput_perGroup_train(data,groupId)
     local nBatch = data:size(1)
     local nFeature= data:size(2)
     local mean= data.new()
     local centered = data.new()
     local output=data.new()
     local sigma=data.new()
     local set_X={}
     
     local eye_nDim
     if groupId ~= self.n_groups then 
       eye_nDim=self.eye_ngroup 
     else
       eye_nDim=self.eye_ngroup_last 
     end 
     
     centered:resizeAs(data)
     output:resizeAs(data)
     mean:mean(data, 1)                        -- E(x) = expectation of x.
     self.running_means[groupId]:mul(1 - self.momentum):add(self.momentum, mean) -- add to running mean
      -- subtract mean
     centered:add(data, -1, mean:expandAs(data))         -- x - E(x)
     sigma:resize(data:size(2),data:size(2))
      
     sigma:addmm(0,sigma,1/nBatch,centered:t(),centered) 
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
        print('----debug IterNorm: running mean and running projection------')
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
 
 
 ------------------------ test mode-----------------------------------------
 
  function updateOutput_perGroup_test(data,groupId)
      local nBatch = data:size(1)
      local nFeature = data:size(2)
      local centered=data.new()
      local output=data.new()
      centered:resizeAs(data):copy(data)
      centered:add(-1,self.running_means[groupId]:view(1,nFeature):expandAs(data))
      
      output:resizeAs(data)
      output:mm(centered,self.running_projections[groupId]) 
     
      return output
  end

---------------------------------------------------------------------------------------
--------------------updateOutput main function-------------------------
----------------------------------------------------------------------------------------


   local nDim=input:size(2)


   assert(nDim  == self.nDim, 'make sure the dimensions of the input is same as the initionazation')
   
   local groups=torch.floor((nDim-1)/self.m_perGroup)+1
    
   self.output=self.output or input.new()
   self.output:resizeAs(input)
   
   self.gradInput=self.gradInput or input.new()
   self.gradInput:resizeAs(input)
   
   self.normalized = self.normalized or input.new() --used for the affine transformation to calculate the gradient
   self.normalized:resizeAs(input)
   -- buffers that are reused
   self.buffer = self.buffer or input.new()
   self.buffer_2 = self.buffer_2 or input.new()


   if self.train == false then
     if self.debug then
       print('----------------IterNorm:test mode***update output***-------------------')
     end
     for i=1,groups do 
        local start_index=(i-1)*self.m_perGroup+1
        local end_index=math.min(i*self.m_perGroup,nDim)      
        self.output[{{},{start_index,end_index}}]=updateOutput_perGroup_test(input[{{},{start_index,end_index}}],i)   
     end
      
   else -- training mode
     
     --------------training mode, initalize the group parameters---------------
      self.sigmas={}
      self.set_Xs={}
      self.centereds={}
      self.whiten_matrixs={}
      if self.debug then
        print('----------------IterNorm:train mode***update output***---------------')
      end
      for i=1,groups do 
         local start_index=(i-1)*self.m_perGroup+1
         local end_index=math.min(i*self.m_perGroup,nDim)      
         self.output[{{},{start_index,end_index}}]=updateOutput_perGroup_train(input[{{},{start_index,end_index}}],i)   
      end
   end
   self.normalized:copy(self.output)
  ------------------------------------------------------------------------ 
  -----------------------scale the output-------------------------------- 
 ------------------------------------------------------------------------   
   
   if self.affine then
      -- multiply with gamma and add beta
       self.output:cmul(self.weight:view(1,nDim):expandAs(self.output))
       self.output:add(self.bias:view(1,nDim):expandAs(self.output))
    end

   collectgarbage()
   return self.output
end

function DecorelateBN_PowerIter_GPU:updateGradInput(input, gradOutput)

  
  -------update the gradInput per Group in train mode------------------------- 

   function Matrix_Pow3(input)
     local b=torch.mm(input, input)
     return torch.mm(b,input) 
   end

   function updateGradInput_perGroup_train_new(gradOutput_perGroup,groupId)
     local  sigma=self.sigmas[groupId]
     local  centered=self.centereds[groupId]
     local  whiten_matrix=self.whiten_matrixs[groupId]
     local  set_X=self.set_Xs[groupId]
     local eye_nDim
     local nBatch = gradOutput_perGroup:size(1) 
     local nFeature = gradOutput_perGroup:size(2) 
     local trace=torch.trace(sigma)
     local sigma_norm=sigma/trace
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
    
    -- print(set_X[1])    
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
     local  running_projection=gradOutput_perGroup:new()
      running_projection=self.running_projections[groupId] --use running projection
  --   local nBatch = gradOutput_perGroup:size(1) 
     self.buffer:resizeAs(gradOutput_perGroup)
     self.buffer:mm(gradOutput_perGroup,running_projection:t())
     return self.buffer
  end
  
  
    
---------------------------------------------------------------------------------------
--------------------updateGradInput main function-------------------------
----------------------------------------------------------------------------------------
  
   assert(input:dim() == 2, 'only mini-batch supported')
   assert(gradOutput:dim() == 2, 'only mini-batch supported')
  
   local nDim=input:size(2)
   local groups=torch.floor((nDim-1)/self.m_perGroup)+1
  
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   if self.affine then
      self.gradInput:cmul(self.weight:view(1,nDim):expandAs(self.gradInput))
   end
   if self.train == false then -- test mode: in which the whitening parameter is fixed (not the function of the input)
      for i=1,groups do 
        local start_index=(i-1)*self.m_perGroup+1
        local end_index=math.min(i*self.m_perGroup,nDim)      
        self.gradInput[{{},{start_index,end_index}}]=updateGradInput_perGroup_test(self.gradInput[{{},{start_index,end_index}}],i)   
      end
      
   else --train mode 
       for i=1,groups do 
         local start_index=(i-1)*self.m_perGroup+1
         local end_index=math.min(i*self.m_perGroup,nDim)
         self.gradInput[{{},{start_index,end_index}}]=updateGradInput_perGroup_train_new(self.gradInput[{{},{start_index,end_index}}],i)   
       end

    end
 
   collectgarbage()

   return self.gradInput
end


function DecorelateBN_PowerIter_GPU:setTrainMode(isTrain)
  if isTrain ~= nil then
      assert(type(isTrain) == 'boolean', 'isTrain has to be true/false')
      self.train = isTrain
  else
    self.train=true  

  end
end

function DecorelateBN_PowerIter_GPU:clearState()
   -- self.buffer:set()
   -- self.buffer_2:set()
   -- self.gradInput:set()
   -- self.f:set()
   -- self.eye_ngroup:set()
   -- self.eye_ngroup_last:set()
   -- self.output:set()
   --  self.centereds=nil

end

function DecorelateBN_PowerIter_GPU:accGradParameters(input, gradOutput, scale)
    if self.affine then
      if self.flag_inner_lr then
        scale = self.scale or 1.0
      else
        scale =scale or 1.0
      end
      self.buffer_2:resizeAs(self.normalized):copy(self.normalized)
      self.buffer_2:cmul(gradOutput)
      self.buffer:sum(self.buffer_2, 1) 
      self.gradWeight:add(scale, self.buffer)
      self.buffer:sum(gradOutput, 1) 
      self.gradBias:add(scale, self.buffer)
   end
end
